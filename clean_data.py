from imagededup.methods import PHash
import os
from tqdm import tqdm
import cv2
from shutil import copyfile
from PIL import Image
import io

os.mkdir('clean_data')

files = os.listdir('SHIFTLabel')
for i in tqdm(range(len(files))):
    try:
      with open('SHIFTLabel/' + files[i], 'rb') as f:
        b = io.BytesIO(f.read())
        Image.open(b)
    except OSError:
      print('SHIFTLabel/' + files[i])
      os.remove('SHIFTLabel/' + files[i])

files = os.listdir('SHIFTLabel')

for i in tqdm(range(len(files))):
    im = cv2.imread('SHIFTLabel/' + files[i], cv2.IMREAD_GRAYSCALE)
    copyfile('SHIFTLabel/' + files[i], 'clean_data/' + files[i])

phasher = PHash()

encodings = phasher.encode_images(image_dir='clean_data/')

duplicates = phasher.find_duplicates(encoding_map=encodings)

data = []
dkeys = list(duplicates.keys())
ind_key = 0
while (True):
    print(len(dkeys))
    data.append(dkeys[ind_key])
    if (len(duplicates[dkeys[ind_key]]) > 0):
        for j in range(len(duplicates[dkeys[ind_key]])):
            try:
                dkeys.remove(duplicates[dkeys[ind_key]][j])
                os.remove('clean_data/' + duplicates[dkeys[ind_key]][j])
            except:
                pass
    ind_key += 1
    if (ind_key >= len(dkeys) - 1):
        break;

print(data)
print(len(data))