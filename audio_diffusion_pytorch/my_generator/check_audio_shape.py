import os
import librosa

directory = '/home/aix23606/jungmin/kaggle_dataset/cat_audio/dataset'

audio_files = [file for file in os.listdir(directory) if file.endswith('.wav')]

print(len(audio_files))

for audio_file in audio_files:
    file_path = os.path.join(directory, audio_file)

    waveform, sample_rate = librosa.load(file_path, sr=None)

    num_channels = waveform.ndim

    print(f"File: {audio_file}")
    print(f"Number of channels: {num_channels}")
    print(f"Sample rate: {sample_rate}")
    print()

