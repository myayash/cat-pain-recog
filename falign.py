import base64
import json
import requests

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import matplotlib.cm as cm
import numpy as np

def conv_img_bs64(img_path):
  print('–'*10)
  print('CONVERTING to BS64 STRING...')
  with open(img_path, 'rb') as img:
    bs64_string = base64.b64encode(img.read()).decode('utf-8')
  #print(f'SHOWING bs64 string:\n{bs64_string}\n')
  return bs64_string

def json_load(img_path, img_bs64_string):
  print('–'*10)
  print('WRITING PAYLOAD...')
  payload = {
    'name': img_path.split('/')[-1],
    'image': f'data:image/jpeg;base64, {img_bs64_string}'
  }
  #print(f'SHOWING PAYLOAD...\n{payload}\n')
  return json.dumps(payload)

def send_img(img_path, url):
  img_bs64_string = conv_img_bs64(img_path)
  request = json_load(img_path, img_bs64_string)

  print('–'*10)
  print('CREATING HEADERS...')
  headers = {'Content-Type': 'application/json'}
  response = requests.post(url, data=request, headers=headers)
  print(f'SHOWING RESPONSE...\n{response}')

  if response.status_code == 200:
    print('–'*10)
    print('IMAGE PROCESSED')
    return response.json()
  else:
    print('–'*10)
    print('IMAGE PROCESSING FAILED')

url = "http://34.165.76.57:6000/landmarks"
image_path = "/Users/macbook/Documents/skripsi_ayas/custom testing images (refs)/feighel no pain 1.png"  
print(f'API URL {url}\nINPUT_PATH{image_path}')

result = send_img(image_path, url)
print('RESPONSE: ')
print(result)

print('PLOTTING...')
image = Image.open(image_path)

fig, ax = plt.subplots()

# define number of distinct animals
num_animals = len(result)
cmap = matplotlib.colormaps['rainbow']

for i, animal_data in enumerate(result):
  print(f'\n[{i}] ANIMAL DATA:\n {animal_data}')
  print(f'\nANIMAL_DATA.ITEMS():\n{animal_data.items()}')
  for animal, details in animal_data.items():
    print(f'\nANIMAL: {animal}')
    print(f'\nDETAILS: {details}')
    color = cmap(i / num_animals +2)  # get a color from colormap
    landmarks = details['landmarks']
    for landmark in landmarks:
      x = landmark['x']
      y = landmark['y']
      ax.scatter(x, y, color=color,  s=5)  # plot the point with the color from the colormap

plt.axis('off')
plt.savefig('plotted_lms.png')
# Show the plot
ax.imshow(image)
plt.show()
