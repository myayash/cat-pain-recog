print(f'IMPORTING MODULES...')
import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms.v2 as transforms
import torchvision.transforms.functional as F
import os
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

from transformers import AutoModelForImageClassification
from falign import conv_img_bs64, json_load, send_img, process_response

def build_detect_cat(model_path):
    print(f'\nLOADING DETECTOR MODEL... ')
    detect_model = AutoModelForImageClassification.from_pretrained(model_path)
    detect_model.eval()
    print('–'*10)
    return detect_model

def detect_cats(image_input_tensor, model):
    print(f'\nDETECTING CATS...')
    with torch.no_grad():
        output = model(image_input_tensor)
        predicted = torch.argmax(output.logits, dim=1)
        print(f'PREDICTED VALUE: {predicted}')

        if predicted.item() == 1:
            print(f'ITS A NOT A CAT!')
            print('–'*10)
            return 'DOG'
        else:
            print(f'ITS A CAT!')
            print('–'*10)
            return 'CAT'

def build_model(pretrained_model_path, device):
    model_name = os.path.basename(pretrained_model_path)
    print(f'\nPAIN RECOGNITION using {model_name} model')

    model = models.resnet18().to(device)
    
    print(f'Modifying FC layer...')
    model.fc = nn.Sequential(
    nn.Linear(512,64),
    nn.ReLU(),
    nn.Dropout(),
    nn.Linear(64,32),
    nn.ReLU(),
    nn.Dropout(),
    nn.Linear(32,1),
    nn.Sigmoid()
)
    
    print('LOADING MODEL...')
    model.load_state_dict(torch.load(pretrained_model_path, map_location=device)['model_state_dict'])
    print(f'Setting model to model.eval()...')
    model.eval()
    print('–'*10)
    return model

def resize_pad(img):
    target_size = (224, 224)
    target_height, target_width = target_size
    
    height, width = img.shape[-2:]
    aspect_ratio = width / height

    if width > height:
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:
        new_height = target_height
        new_width = int(target_height *  aspect_ratio)
    
    img_resized = F.resize(img, (new_height, new_width))
    print(f'RESIZED IMAGE TYPE: {type(img_resized)}')
    print(f'RESIZED IMAGE SIZE: {img_resized.size()}')

    print(f'\nCreating padding background image...')
    padding = Image.new('RGB', target_size, 0)
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    
    print(f'PADDING TYPE: {type(padding)}')
    print(f'PADDING SIZE: {padding.size}')
    
    print(f'\nConverting resized img to PIL...')
    img_resized = F.to_pil_image(img_resized)
    print(f'RESIZED IMAGE TYPE: {type(img_resized)}')
    print(f'RESIZED IMAGE SIZE: {img_resized.size}')

    print(f'\nPasting image to padding...')
    padding.paste(img_resized, (paste_x, paste_y))
    print(f'PASTED IMAGE TYPE {type(padding)}')
    print(f'PASTED IMAGE SIZE {padding.size}')

    print(f'Showing resized/padded image...')
    
    return F.to_tensor(padding)

def preprocess_img(image_input):
    image_input = Image.open(image_input)
    fig, ax = plt.subplots()
    print('–'*10)
    print(f'\nShowing image...')
    ax.imshow(image_input)
    ax.set_title('Before Preprocessing')
    ax.axis('off')

    print(f'IMAGE TYPE: {type(image_input)}')
    print(f'IMAGE SIZE: {image_input.size}')
    print(f'IMAGE MODE: {image_input.mode}')
    print(f'\nProcessing image...')

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.ToImage(),
        transforms.Lambda(lambda img: resize_pad(img)),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean, std)
    ])
    

    image_tensor = transform(image_input).unsqueeze(0).to(device)

    print(f'\nTRANSFORMED IMAGE TYPE: {type(image_tensor)}')
    print(f'TRANSFORMED IMAGE SIZE: {image_tensor.size()}')

    image_tensor = image_tensor.squeeze()
    print(f'SQUEEZED TENSOR SIZE: {image_tensor.size()}')

    t_c, t_h, t_w = image_tensor.shape
    denorm_image = torch.empty_like(image_tensor)

    for c in range(t_c):
        for h in range(t_h):
            for w in range(t_w):
                denorm_image[c, h, w] = image_tensor[c, h, w] * std[c] + mean[c]

    denorm_np = denorm_image.cpu().detach().numpy()
    denorm_np = denorm_np.transpose((1,2,0))

    fig_t, ax_t = plt.subplots()
    ax_t.imshow(denorm_np)
    ax_t.set_title('Tensor Image')
    ax_t.axis('off')

    plt.show()

    print('–'*10)

    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor

def predict(image_input_tensor, model):
    with torch.no_grad():
        print('\nANALYZING...')
        output = model(image_input_tensor)
        print(f'OUTPUT logit: {output}')
        print(f'OUTPUT shape: {output.shape}')
          
        prediction = output.squeeze(1) >= thr
        print(f'\nOUTPUT AFTER thr: {prediction}')
        print(f'OUTPUT shape AFTER squeeze: {output.squeeze(1).shape}')
        
        confidence = torch.sigmoid(output.squeeze(1))
        if prediction == True:
            print('–'*10)
            return print(f'\nPrediksi: ucing ATIT T__T, {confidence.item()*100:.4f}% yakin')
        else:
            print('–'*10)
            return print(f'\nPrediksi: ucing CEHAT n__n, {(1-confidence.item())*100:.4f}% yakin')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
api_url = "http://34.165.76.57:6000/landmarks"
model_path = '/Users/macbook/Documents/projects/cat pain recog/models/Copy of rn18 adamW FL final.pth'
detection_model_path = '/Users/macbook/Documents/projects/cat pain recog/models/detector'
image_path = input('Enter image file (absolute path): ')
thr = 0.4

# preprocess img further
image_tensor = preprocess_img(image_path)

detect_model = build_detect_cat(detection_model_path)
_ = detect_cats(image_tensor, detect_model)

if _ == 'CAT':

    # get landmarks via api
    result = send_img(image_path, api_url)
    print(f'RESPONSE:\n {result}')
    # process image with landmarks
    cropped_image = process_response(result, image_path)   
    model = build_model(model_path, device)
    predict(image_tensor, model)
else:
    print('\nGET A CAT IMAGE')





