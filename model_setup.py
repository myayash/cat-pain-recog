import torch
import torchvision.models as models
import torch.nn as nn
import os
from transformers import AutoModelForImageClassification


def build_detect_cat(model_path, verbose=False):
    print(f'\nLOADING DETECTOR MODEL... ')
    detect_model = AutoModelForImageClassification.from_pretrained(model_path)
    detect_model.eval()
    print('–'*10)
    return detect_model

def detect_cats(image_input_tensor, model, verbose=False):
    print(f'\nDETECTING CATS...')
    with torch.no_grad():
        output = model(image_input_tensor)
        predicted = torch.argmax(output.logits, dim=1)
        if verbose == True:
            print(f'PREDICTED VALUE: {predicted}')

        if predicted.item() == 1:
            print(f'ITS A NOT A CAT!')
            print('–'*10)
            return 'DOG'
        else:
            print(f'ITS A CAT!')
            print('–'*10)
            return 'CAT'

def build_model(pretrained_model_path, device='cpu', verbose=False):
    model_name = os.path.basename(pretrained_model_path)
    print(f"\nPain recognition using MODEL '{model_name}'")

    model = models.resnet18().to(device)
    
    if verbose == True:
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

def predict(image_input_tensor, model, thr=0.4, verbose=False):
    with torch.no_grad():
        print('\nANALYZING...')
        output = model(image_input_tensor)
        if verbose == True:
            print(f'OUTPUT logit: {output}')
            print(f'OUTPUT shape: {output.shape}')
          
        prediction = output.squeeze(1) >= thr
        
        if verbose == True:
            print(f'\nOUTPUT AFTER thr: {prediction}')
            print(f'OUTPUT shape AFTER squeeze: {output.squeeze(1).shape}')
        
        confidence = torch.sigmoid(output.squeeze(1))

        if prediction == True:
            print('–'*10)
            print(f'\nPrediksi: ucing ATIT T__T, {confidence.item()*100:.4f}% yakin!')
            return 'pain'
        else:
            print('–'*10)
            print(f'\nPrediksi: ucing CEHAT n__n, {(1-confidence.item())*100:.4f}% yakin!')
            return 'no pain'


