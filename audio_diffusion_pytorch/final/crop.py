import os
from pydub import AudioSegment

input_wav_directory = '/home/aix23606/jungmin/audio_datasets/classic_youtube'
wav_file = '/home/aix23606/jungmin/audio_datasets/classic_youtube/classic_audio5.wav'
output_cropped_directory = '/home/aix23606/jungmin/audio_dataset/classic_audio_cropped'

os.makedirs(output_cropped_directory, exist_ok=True)

crop_duration = 10000

audio = AudioSegment.from_wav(wav_file)
num_segments = len(audio) // crop_duration
for i in range(num_segments):
    start_time = i * crop_duration
    end_time = (i + 1) * crop_duration
    cropped_audio = audio[start_time:end_time]
    cropped_path = os.path.join(output_cropped_directory, f"{os.path.splitext('classic_audio5')[0]}_{i + 1}.wav")

    cropped_audio.export(cropped_path, format="wav")

    print(f"Cropped {'classic_audio5'} segment {i + 1} to {cropped_path}")