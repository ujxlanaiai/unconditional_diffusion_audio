import os
import librosa
import numpy as np

dir = '/home/aix23606/jungmin/audio_dataset/classic_audio_cropped'

for file in os.listdir(dir):
    file_path = os.path.join(dir, file)

    try:
        waveform, sample_rate = librosa.load(file_path, sr=None)
    except (FileNotFoundError, RuntimeError, EOFError) as e:
        # Handle specific exceptions that may occur when loading audio files
        print(f"Error loading file {file}: {e}")
        os.remove(file_path)
        print(f"File {file} removed due to loading error.")
        continue

    if waveform is None or len(waveform) == 0 or np.mean(waveform) == 0:
        os.remove(file_path)
        print(f"File {file} removed because it has invalid waveform data.")

print("Processing complete.")
