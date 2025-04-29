import torch
import torchvision.transforms.v2 as transforms
import torchvision.transforms.functional as F
from PIL import Image


def resize_pad(img, verbose=False):
    target_size = (224, 224)
    target_height, target_width = target_size
    
    height, width = img.shape[-2:]
    aspect_ratio = width / height

    if width > height:
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:
        new_height = target_height
        new_width = int(target_height / aspect_ratio)
    
    img_resized = F.resize(img, (new_height, new_width))

    if verbose == True:
        print(f'RESIZED IMAGE TYPE: {type(img_resized)}')
        print(f'RESIZED IMAGE SIZE: {img_resized.size()}')

        print(f'\nCreating padding background image...')

    padding = Image.new('RGB', target_size, 0)
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2

    if verbose == True: 
        print(f'PADDING TYPE: {type(padding)}')
        print(f'PADDING SIZE: {padding.size}')
    
        print(f'\nConverting resized img to PIL...')

    img_resized = F.to_pil_image(img_resized)
    if verbose == True:
        print(f'RESIZED IMAGE TYPE: {type(img_resized)}')
        print(f'RESIZED IMAGE SIZE: {img_resized.size}')

        print(f'\nPasting image to padding...')

    padding.paste(img_resized, (paste_x, paste_y))

    if verbose == True:
        print(f'PASTED IMAGE TYPE {type(padding)}')
        print(f'PASTED IMAGE SIZE {padding.size}')

    print(f'Showing resized/padded image... {padding.show()}')
    
    scaled_image_tensor = F.to_tensor(padding)

    return scaled_image_tensor

def preprocess_img(image_input, device='cpu', verbose=False):
    image = Image.open(image_input)

    print('–'*10)
    print(f'\nShowing image... {image.show()}')
    
    if verbose == True:
        print(f'IMAGE TYPE: {type(image)}')
        print(f'IMAGE SIZE: {image.size}')
        print(f'IMAGE MODE: {image.mode}')

    print(f'\nProcessing image...')

    transform = transforms.Compose([
        transforms.ToImage(),
        transforms.Lambda(lambda img: resize_pad(img)),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_tensor = transform(image).unsqueeze(0).to(device)
    
    print('–'*10)

    return image_tensor


