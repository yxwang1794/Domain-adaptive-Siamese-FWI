import torch
import numpy as np
import deepwave
from Module26Gaussian import *
import math
import scipy.ndimage
import matplotlib.pyplot as plt
from gaussian_blur import LowFrequencyExtractor
from domain_discriminator import DomainDiscriminator2D

def set_seed(seed=99):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(99)

freq = 10
dx = 20
dt = 0.001
nt = 2000
nz = 100
nx = 401
num_shots = 40
source_spacing = 10
device = torch.device('cuda:0')
true_model = np.load('overthrust.npy')
model_init = scipy.ndimage.gaussian_filter(true_model,sigma=20)
plt.subplot(1,2,1)
plt.imshow(true_model,cmap='seismic',vmin=600,vmax=6000)
plt.title("True Model")
plt.subplot(1,2,2)
plt.imshow(model_init,cmap='seismic',vmin=600,vmax=6000)
plt.title("Initial Model")
plt.show()
true_model = torch.tensor(true_model,dtype=torch.float32, device=device)
model_init = torch.tensor(model_init,dtype=torch.float32, device=device)

source_l = torch.zeros(num_shots,1,2)
source_l[:, 0, 1] = (1.0+torch.arange(num_shots)).float() * source_spacing
receiver_l = torch.zeros(num_shots, 400,2)
receiver_l[0, :, 1] = torch.arange(400).float() * 1
receiver_l[:, :, 1] = receiver_l[0, :, 1].repeat(num_shots, 1)
source_amplitude_true = deepwave.wavelets.ricker(freq, nt, dt, 0.3).reshape(1, 1, -1)
source_amplitude_assumed = deepwave.wavelets.ricker(20, nt, dt, 0.3).reshape(1, 1, -1)
source_func_true = source_amplitude_true.repeat(num_shots,1,1)
source_func_assumed = source_amplitude_assumed.repeat(num_shots,1,1)
plt.plot(source_amplitude_true.detach().cpu().numpy()[0,0,:],label="true source wavelet")
plt.plot(source_amplitude_assumed.detach().cpu().numpy()[0,0,:],label="assumed source wavelet")
plt.legend()
plt.show()

receiver_amplitudes_true = deepwave.scalar(
    true_model.to(device),
    dx,
    dt,
    source_amplitudes=source_func_true.to(device),
    source_locations=source_l.to(device),
    receiver_locations=receiver_l.to(device)
)[-1]

model_fwi = model_init.clone().detach().to(device).requires_grad_(True)
opt_fwi = torch.optim.Adam([{'params':[model_fwi], 'lr':30, 'weight_decay':0e-11}])
model_fwi = model_fwi.to(device)
model_fwi.requires_grad = True
num_batches = 8
num_shots_per_batch = int(num_shots / num_batches)
num_epochs = 100

model_dasia = model_init.clone().detach().to(device).requires_grad_(True)
opt_dasia = torch.optim.Adam([{'params':[model_dasia], 'lr':30, 'weight_decay':0e-11}])
model_dasia = model_dasia.to(device)
model_dasia.requires_grad = True
DASiamese = FrequencyAttentionFusion(5)
DASiamese = DASiamese.to(device)
optim_DAsiamese = optim.Adam(DASiamese.parameters(),lr=10e-4, betas=(0.5, 0.99), \
                    eps=1e-6, weight_decay=0)
low_extractor = LowFrequencyExtractor(pyramid_levels=2).to(device)
disc = DomainDiscriminator2D(in_ch=5, base=32, p=0.01).to(device)
optim_D = optim.AdamW(disc.parameters(), lr=1e-4, weight_decay=1e-4)
criterion_dom = nn.CrossEntropyLoss()

for epoch in range(100):
    # ===================== Stage A：Train Discriminator（Freeze Siamese） =====================
    DASiamese.eval()
    for p in DASiamese.parameters(): p.requires_grad_(False)
    model_dasia.requires_grad_(False)

    disc.train()
    for p in disc.parameters():
        p.requires_grad_(True)

    lossD_epoch, accD_epoch, totD = 0.0, 0, 0
    if epoch%1==0:
        for it in range(num_batches):
            optim_D.zero_grad()

            batch_src_amps = source_amplitude_assumed.repeat(num_shots_per_batch, 1, 1)
            batch_rcv_true = receiver_amplitudes_true[it::num_batches].to(device)
            batch_x_s = source_l[it::num_batches].to(device)
            batch_x_r = receiver_l[it::num_batches].to(device)

            with torch.no_grad():
                batch_rcv_pred = deepwave.scalar(
                    model_dasia.to(device), dx, dt,
                    source_amplitudes=batch_src_amps.to(device),
                    source_locations=batch_x_s.to(device),
                    receiver_locations=batch_x_r.to(device)
                )[-1]

            u_obs = batch_rcv_true / torch.sqrt(torch.mul(batch_rcv_true, batch_rcv_true).sum())
            u_pred = batch_rcv_pred / torch.sqrt(torch.mul(batch_rcv_pred, batch_rcv_pred).sum())

            in_1 = low_extractor(u_obs.unsqueeze(0))
            in_2 = low_extractor(u_pred.unsqueeze(0))

            with torch.no_grad():
                f_obs, f_pred = DASiamese(in_1, in_2)  # [5,1,H,W] x2

            feats = torch.cat([f_obs.detach(), f_pred.detach()], dim=0)  # [10,1,H,W]
            labelsD = torch.cat([
                torch.zeros(f_obs.size(0), dtype=torch.long, device=device),  # 0=obs
                torch.ones(f_pred.size(0), dtype=torch.long, device=device)  # 1=pred
            ], dim=0)

            logitsD = disc(feats)  # [10,2]

            lossD = criterion_dom(logitsD, labelsD)
            lossD.backward()
            optim_D.step()

            lossD_epoch += lossD.item()
            predD = logitsD.argmax(1)
            accD_epoch += (predD == labelsD).sum().item()
            totD += labelsD.numel()

        print(
            f"[D] Epoch {epoch + 1}/{num_epochs} | Loss {lossD_epoch / num_batches:.4f} | Acc {100 * accD_epoch / max(1, totD):.2f}%")

    # ===================== Stage B：DA-Siamese Training =====================
    for p in disc.parameters(): p.requires_grad_(False)
    disc.eval()

    DASiamese.train()
    for p in DASiamese.parameters(): p.requires_grad_(True)
    model_dasia.requires_grad_(True)

    epoch_total = 0.0
    for it in range(num_batches):
        opt_dasia.zero_grad()
        optim_DAsiamese.zero_grad()     # Siamese

        batch_src_amps = source_amplitude_assumed.repeat(num_shots_per_batch, 1, 1)
        batch_rcv_true = receiver_amplitudes_true[it::num_batches].to(device)
        batch_x_s = source_l[it::num_batches].to(device)
        batch_x_r = receiver_l[it::num_batches].to(device)

        batch_rcv_pred = deepwave.scalar(
            model_dasia.to(device), dx, dt,
            source_amplitudes=batch_src_amps.to(device),
            source_locations=batch_x_s.to(device),
            receiver_locations=batch_x_r.to(device)
        )[-1]

        u_obs  = batch_rcv_true / torch.sqrt(torch.mul(batch_rcv_true, batch_rcv_true).sum())
        u_pred = batch_rcv_pred / torch.sqrt(torch.mul(batch_rcv_pred, batch_rcv_pred).sum())



        in_1 = low_extractor(u_obs.unsqueeze(0))
        in_2 = low_extractor(u_pred.unsqueeze(0))

        f_obs, f_pred = DASiamese(in_1, in_2)     # [5,1,H,W] x2

        loss_main = F.pairwise_distance(f_obs, f_pred, keepdim=True).mean()

        logits_obs  = disc(f_obs)
        logits_pred = disc(f_pred)
        labels_obs  = torch.zeros(f_obs.size(0), dtype=torch.long, device=device)
        labels_pred = torch.ones(f_pred.size(0), dtype=torch.long, device=device)
        loss_dom_true = 0.5 * (criterion_dom(logits_obs, labels_obs) + criterion_dom(logits_pred, labels_pred))

        loss = loss_main - loss_dom_true*0.05
        loss.backward()
        opt_dasia.step()
        optim_DAsiamese.step()

        epoch_total += loss.item()

    print(f"[ADV] Epoch {epoch+1}/{num_epochs} | Total (main + tv - λ*dom) = {epoch_total/num_batches:.6f}")


plt.imshow(model_dasia.cpu().detach().numpy(),cmap='seismic',vmin=600,vmax=6000)
plt.show()
np.save('model_dasia.npy',model_dasia.cpu().detach().numpy())