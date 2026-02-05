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

model_sia = model_init.clone().detach().to(device).requires_grad_(True)
opt_sia = torch.optim.Adam([{'params':[model_sia], 'lr':30, 'weight_decay':0e-11}])
model_sia = model_sia.to(device)
model_sia.requires_grad = True
Siamese = FrequencyAttentionFusion(5)
Siamese = Siamese.to(device)
optim_siamese = optim.Adam(Siamese.parameters(),lr=10e-4, betas=(0.5, 0.99), \
                    eps=1e-6, weight_decay=0)
low_extractor = LowFrequencyExtractor(pyramid_levels=2).to(device)
num_batches = 8
num_shots_per_batch = int(num_shots / num_batches)
for epoch in range(100):
    epoch_loss = 0.0

    for it in range(num_batches):
        opt_sia.zero_grad()
        batch_src_amps = source_amplitude_assumed.repeat(num_shots_per_batch, 1, 1)
        batch_rcv_amps_true = receiver_amplitudes_true[it::num_batches].to(device)

        batch_x_s = source_l[it::num_batches].to(device)
        batch_x_r = receiver_l[it::num_batches].to(device)

        batch_rcv_amps_pred = deepwave.scalar(
            model_sia.to(device),
            dx,
            dt,
            source_amplitudes=batch_src_amps.to(device),
            source_locations=batch_x_s.to(device),
            receiver_locations=batch_x_r.to(device)
        )[-1]

        u_obs = batch_rcv_amps_true / torch.sqrt(torch.mul(batch_rcv_amps_true, batch_rcv_amps_true).sum())
        u_pred = batch_rcv_amps_pred / torch.sqrt(torch.mul(batch_rcv_amps_pred, batch_rcv_amps_pred).sum())

        u_obs = low_extractor(u_obs.unsqueeze(0))
        u_pred = low_extractor(u_pred.unsqueeze(0))
        output1, output2 = Siamese(u_obs, u_pred)
        loss = F.pairwise_distance(output1, output2, keepdim=True).mean()
        epoch_loss += loss.item()
        loss.backward()
        opt_sia.step()
        optim_siamese.step()
    print('current epoch: %d,loss: %.6f' % (epoch, loss))

plt.imshow(model_sia.cpu().detach().numpy(),cmap='seismic',vmin=600,vmax=6000)
plt.show()