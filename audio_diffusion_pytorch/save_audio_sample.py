from my_generator.uncongen_suc import sample
import torch
import torchaudio

sample_np = sample.squeeze().cpu().numpy()
sample_rate = 44100

print(sample_np.shape)
print('Saving audio')

torchaudio.save("guitar_3.wav", torch.tensor(sample_np), sample_rate, channels_first=True)

print('Save complete')