import os
import pretty_midi
from midi2audio import FluidSynth
from pydub import AudioSegment

input_midi_directory = '/home/aix23606/jungmin/kaggle_dataset/piano_midi'
output_wav_directory = '/home/aix23606/jungmin/kaggle_dataset/piano_audio'
output_cropped_directory = '/home/aix23606/jungmin/kaggle_dataset/piano_audio_cropped'

os.makedirs(output_wav_directory, exist_ok=True)
os.makedirs(output_cropped_directory, exist_ok=True)

fs = FluidSynth(sound_font='/home/aix23606/jungmin/Yamaha_MA2_soundfont/Yamaha_MA2.sf2')

crop_duration = 10000

for filename in os.listdir(input_midi_directory):
    if filename.endswith(".mid"):
        midi_path = os.path.join(input_midi_directory, filename)
        midi_data = pretty_midi.PrettyMIDI(midi_path)

        wav_path = os.path.join(output_wav_directory, os.path.splitext(filename)[0] + ".wav")

        # Set SoundFont explicitly to avoid default SoundFont errors
        fs.midi_to_audio(midi_path, wav_path)

        print(f"Converted {filename} to {wav_path}")

        audio = AudioSegment.from_wav(wav_path)

        num_segments = len(audio) // crop_duration

        for i in range(num_segments):
            start_time = i * crop_duration
            end_time = (i + 1) * crop_duration
            cropped_audio = audio[start_time:end_time]  # Corrected slicing syntax

            cropped_path = os.path.join(output_cropped_directory, f"{os.path.splitext(filename)[0]}_{i+1}.wav")

            cropped_audio.export(cropped_path, format="wav")

            print(f"Cropped {filename} segment {i+1} to {cropped_path}")