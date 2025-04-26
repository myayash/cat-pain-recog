import base64
import json
import requests

def conv_img_bs64(img_path):
  with open(img_path, 'rb') as img:
    bs64_string = base64.b64encode(img.read()).decode('utf-8')
  return bs64_string

def json_load(img_path, img_bs64_string):
  payload = {
    'name': img_path.split('/')[-1],
    'image': f'data:image/jpeg;base64, {img_bs64_string}'
  }
  return json.dumps(payload)

def send_img(img_path, url):
  img_bs64_string = conv_img_bs64(img_path)
  request = json_load(img_path, img_bs64_string)

  headers = {'Content-Type': 'application/json'}
  response = requests.post(url, data=request, headers=headers)

  if response.status_code == 200:
    print('IMAGE PROCESSED')
    print('RESPONSE: ', response.json())
    return response.json()
  else:
    print('IMAGE PROCESSING FAILED')

#@title get cat image

cat_url = "https://cdn.pixabay.com/photo/2021/01/29/19/28/arctic-wolf-5961985_1280.jpg" # @param {type:"string"}

# Send a GET request to the URL
response = requests.get(cat_url)

# Check if the request was successful (status code 200)
if response.status_code == 200:
  # Open a file in binary write mode and save the image content
  with open("image.jpg", "wb") as f:
    f.write(response.content)
  print("Cat image downloaded successfully!")
else:
  print("Failed to download the cat image.")

url = "http://34.165.76.57:6000/landmarks"
image_path = "/Users/macbook/Documents/skripsi_ayas/custom testing images (refs)/feighel no pain 1.png"  # Replace with the actual path to your image

# Send the image for processing
result = send_img(image_path, url)

print(result)
