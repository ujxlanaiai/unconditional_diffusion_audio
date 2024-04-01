from audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler
import torch
import os

# GPU connection

device = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

model = DiffusionModel(
    net_t=UNetV0, # The model type used for diffusion (U-Net V0 in this case)
    in_channels=2, # U-Net: number of input/output (audio) channels
    channels=[8, 32, 64, 128, 256, 512, 512, 1024, 1024], # U-Net: channels at each layer
    factors=[1, 4, 4, 4, 2, 2, 2, 2, 2], # U-Net: downsampling and upsampling factors at each layer
    items=[1, 2, 2, 2, 2, 2, 2, 4, 4], # U-Net: number of repeating items at each layer
    attentions=[0, 0, 0, 0, 0, 1, 1, 1, 1], # U-Net: attention enabled/disabled at each layer
    attention_heads=8, # U-Net: number of attention heads per attention item
    attention_features=64, # U-Net: number of attention features per attention item
    diffusion_t=VDiffusion, # The diffusion method used
    sampler_t=VSampler # The diffusion sampler used
)

model = model.to(device)

print('Successful moving model to device')

# Train model with audio waveforms
print('Start training model')
audio = torch.randn(1, 2, 2**18).to(device) # [batch_size, in_channels, length]
print('Start calculating loss')
print(audio.shape)

loss = model(audio)
print('Start backward')
loss.backward()
print('Done training')

# Turn noise into new audio sample with diffusion
print('Turning noise into new audio sample')
noise = torch.randn(1, 2, 2**18).to(device) # [batch_size, in_channels, length]
sample = model.sample(noise, num_steps=100) # Suggested num_steps 10-100