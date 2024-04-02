from my_generator.train_val_plot import sample
import torch
import torchaudio

sample_np = sample.squeeze().cpu().numpy()
sample_rate = 44100

print(sample_np.shape)
print('Saving audio')

torchaudio.save("piano_4.wav", torch.tensor(sample_np), sample_rate)

print('Save complete')