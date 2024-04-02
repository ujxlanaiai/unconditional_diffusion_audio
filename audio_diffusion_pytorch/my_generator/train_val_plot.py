import numpy as np
from audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler
import torch
import os
from torch.utils.data import Dataset, DataLoader
import librosa
from tqdm import tqdm
import matplotlib.pyplot as plt

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

batch_size = 8
target_length = 2**18

dataset = AudioDataset(directory='/home/aix23606/jungmin/kaggle_dataset/guitar_audio', target_length=target_length)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

num_epochs = 50
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()

print('Ready to train...')

device = "cuda" if torch.cuda.is_available() else "cpu"

if (device == "cuda"):
    print("Successfully GPU running")
else:
    print("CPU running")

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

model = model.to(device)
print('Successfully moved model to device')

train_losses = []
val_losses = []

for epoch in range(num_epochs):
    total_train_loss = 0.0
    total_val_loss = 0.0

    # Training
    model.train()
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
        batch = batch.squeeze(dim=2)
        batch = batch.to(device)
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, batch)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    # Validation
    model.eval()
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
            batch = batch.squeeze(dim=2)
            batch = batch.to(device)
            output = model(batch)
            loss = criterion(output, batch)
            total_val_loss += loss.item()

    train_loss = total_train_loss / len(train_dataloader)
    val_loss = total_val_loss / len(val_dataloader)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}")

# Plotting the loss curve
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs + 1), val_losses, label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig('cat3_loss_plot.png')  # Save the plot as a PNG file
plt.show()

print('Training Completed')

print('Turning noise into new audio sample')
# noise = torch.randn(8, 1, 2**18).to(device)
# sample = model.sample(noise, num_steps=77)

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
