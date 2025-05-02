import os
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

def process_response(result, image_path):

    num_animals = len(result)
    cmap = matplotlib.colormaps['rainbow']

    image = Image.open(image_path)

    fig, ax = plt.subplots()

    for i, animal_data in enumerate(result):
        print(f'\n[{i}] ANIMAL DATA:\n {animal_data}')
        print(f'\nANIMAL_DATA.ITEMS():\n {animal_data.items()}')
        
        color = cmap(i/num_animals+2)

        for animal, details in animal_data.items():
            print(f'\nShowing ANIMAL: {animal}\nDETAILS: {details}')

            landmarks = details['landmarks']
            for landmark in landmarks:
                x = landmark['x']
                y = landmark['y']
                ax.scatter(x, y, color=color, s=5)

        ax.set_title('Original')
        ax.axis('off')
        ax.imshow(image)
        bbox = details['bbox']
        pul = bbox.get('pul')
        pbr = bbox.get('pbr')

        if pul and pbr:
            bbox_x1 = pul.get('x')
            bbox_y1 = pul.get('y')
            bbox_x2 = pbr.get('x')
            bbox_y2 = pbr.get('y')

            if all([isinstance(c, (int,float)) for c in [bbox_x1, bbox_y1, bbox_x2, bbox_y2]]):
                bbox_x1, bbox_y1, bbox_x2, bbox_y2 = int(bbox_x1), int(bbox_y1), int(bbox_x2), int(bbox_y2)

                bbox_w = bbox_x2 - bbox_x1
                bbox_h = bbox_y2 - bbox_y1

                print(f"\nPROCESSING BBOX for DATA [{i}]")

                rect = matplotlib.patches.Rectangle(
                        (bbox_x1, bbox_y1),
                        bbox_w,
                        bbox_h,
                        linewidth=2,
                        edgecolor=color,
                        facecolor='none',
                        label=f'{i} DATA BBOX'
                        )
                ax.add_patch(rect)

                crop_box = (bbox_x1, bbox_y1, bbox_x2, bbox_y2)
                try:
                    cropped_image = image.crop(crop_box)
                    print(f'Showing CROPPED image...')
                    fig_c, ax_c = plt.subplots()

                    ax_c.axis('off')
                    ax_c.set_title('Cropped')
                    ax_c.imshow(cropped_image)

                    landmarks = details['landmarks']
                    for landmark in landmarks:
                        x = landmark['x'] - bbox_x1
                        y = landmark['y'] - bbox_y1
                        ax_c.scatter(x, y, color=color, s=10)

                    plt.show()

                except Exception as e:
                    print(f'Error cropping... ERROR {e}')
            else:
                print('BBOX NOT VALID NUMBERS')
        else:
            print("BBOX MISSING 'pul' or 'pbr' ")

    return cropped_image




