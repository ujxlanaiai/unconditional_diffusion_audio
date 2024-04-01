from audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler
import torch
import torchaudio
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

if (device == "cuda"):
    print("Successfully GPU running")
else:
    print("CPU running")

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
model_path = '/home/aix23606/jungmin/audio-diffusion-pytorch/audio_diffusion_pytorch/saved_model/model4.pth'

loaded_model = DiffusionModel(
    net_t=UNetV0,
    in_channels=1,
    channels=[8, 32, 64, 128, 256, 512, 512, 1024, 1024],
    factors=[1, 4, 4, 4, 2, 2, 2, 2, 2],
    items=[1, 2, 2, 2, 2, 2, 2, 4, 4],
    attentions=[0, 0, 0, 0, 0, 1, 1, 1, 1],
    attention_heads=8,
    attention_features=64,
    diffusion_t=VDiffusion,
    sampler_t=VSampler
)

loaded_model.load_state_dict(torch.load(model_path))
print('Model Loaded')

batch_size = 8

noise = torch.randn(20, 1, 2**18).to(device) # [batch_size, in_channels, length]
loaded_model.to(device)

print('Ready to Generate')

with torch.no_grad():
    sample = loaded_model.sample(noise, num_steps=10)  # Suggested num_steps 10-100

sample_np = sample.squeeze().cpu().numpy()

sample_rate = 44100
print(sample_np.shape)

print('Saving audio')

torchaudio.save("piano_1.wav", torch.tensor(sample_np), sample_rate, channels_first=True)

print('Save complete')