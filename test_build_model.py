from model_setup import build_detect_cat, build_model
from transformers.models.resnet import modeling_resnet 
from torch import device, cuda
from torchvision.models import resnet

device = device('cuda' if cuda.is_available() else 'cpu')
MODEL_PATH = "./cat-pr-rn18-adamW-FL.pth"
DETECTION_MODEL_PATH = "./detector"


def test_model_build():
    detect_model = build_detect_cat(DETECTION_MODEL_PATH)
    model = build_model(MODEL_PATH, device=device)

    assert type(detect_model) is modeling_resnet.ResNetForImageClassification 
    assert type(model) is resnet.ResNet



