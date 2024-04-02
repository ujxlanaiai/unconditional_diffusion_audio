import numpy as np
from audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler

import numpy as np

import torch
import os
from torch.utils.data import Dataset, DataLoader
import librosa
from tqdm import tqdm
import wandb
wandb.init(project='diffusionMusic')


class AudioDataset(Dataset):
    def __init__(self, directory, target_length=None):
        self.directory = directory
        self.audio_files = [file for file in os.listdir(directory) if file.endswith('.wav')]
        self.target_length = target_length

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_file = os.path.join(self.directory, self.audio_files[idx])
        waveform, sample_rate = librosa.load(audio_file, sr=None)

        waveform = (waveform - np.mean(waveform)) / np.std(waveform)

        if self.target_length is not None:
            if len(waveform) < self.target_length:
                pad_width = self.target_length - len(waveform)
                waveform = np.pad(waveform, (0, pad_width), mode='constant')
            else:
                waveform = waveform[:self.target_length]

        waveform = waveform[np.newaxis, np.newaxis, :]

        return torch.tensor(waveform, dtype=torch.float32)

model = DiffusionModel(
    net_t=UNetV0,  # The model type used for diffusion (U-Net V0 in this case)
    in_channels=1,  # U-Net: number of input/output (audio) channels
    channels=[8, 32, 64, 128, 256, 512, 512, 1024, 1024],  # U-Net: channels at each layer
    factors=[1, 4, 4, 4, 2, 2, 2, 2, 2],  # U-Net: downsampling and upsampling factors at each layer
    items=[1, 2, 2, 2, 2, 2, 2, 4, 4],  # U-Net: number of repeating items at each layer
    attentions=[0, 0, 0, 0, 0, 1, 1, 1, 1],  # U-Net: attention enabled/disabled at each layer
    attention_heads=8,  # U-Net: number of attention heads per attention item
    attention_features=64,  # U-Net: number of attention features per attention item
    diffusion_t=VDiffusion,  # The diffusion method used
    sampler_t=VSampler  # The diffusion sampler used
)

batch_size = 8
target_length = 2**18

dataset = AudioDataset(directory='/home/aix23606/jungmin/kaggle_dataset/piano_audio_cropped', target_length=target_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

for batch in dataloader:
    print("Batch shape:", batch.shape)
    break  # Print the shape of the first batch only

num_epochs = 50
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

print('Ready to train...')

# Start training

device = "cuda" if torch.cuda.is_available() else "cpu"

if (device == "cuda"):
    print("Successfully GPU running")
else:
    print("CPU running")

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

model = model.to(device)
print('Successfully moved model to device')

for epoch in range(num_epochs):
    total_loss = 0.0
    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):

        #print("Batch shape before:", batch.shape)  # Print batch shape before processing
        batch = batch.squeeze(dim=2)  # Remove the extra dimension
        #print("Batch shape after:", batch.shape)  # Print batch shape after processing

        batch = batch.to(device)
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        wandb.log({'epoch': epoch, 'loss': loss})
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(dataloader)}")
     
print('Training Completed')
wandb.finish()
print('Turning noise into new audio sample')
noise = torch.randn(8, 1, 2**18).to(device) # [batch_size, in_channels, length]
sample = model.sample(noise, num_steps=77) # Suggested num_steps 10-100

# save model parameter

model_dir = '/home/aix23606/jungmin/audio-diffusion-pytorch/audio_diffusion_pytorch/saved_model'
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, 'model8.pth')

# Save the entire model
try:
    torch.save(model.state_dict(), model_path)
    print(f"Model saved successfully at {model_path}")
except Exception as e:
    print(f"Error occurred while saving the model: {e}")
