import os

directory = '/home/aix23606/jungmin/audio_dataset/classic_audio_cropped'

i = 0
# for files in os.listdir(directory):
#
#     if 'audio4' in files:
#         file_path = os.path.join(directory, files)
#         if os.path.isfile(file_path):
#             os.remove(file_path)

for files in os.listdir(directory):
    i += 1

print("Total files: ", i)