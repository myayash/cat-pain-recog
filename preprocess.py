import os

import matplotlib
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as F
import torchvision.transforms.v2 as transforms
from PIL import Image

matplotlib.use("Agg")  # Use non-GUI backend


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
        new_width = int(target_height * aspect_ratio)

    img_resized = F.resize(img, (new_height, new_width))

    if verbose == True:
        print(f"RESIZED IMAGE TYPE: {type(img_resized)}")
        print(f"RESIZED IMAGE SIZE: {img_resized.size()}")

        print("\nCreating padding background image...")

    padding = Image.new("RGB", target_size, 0)
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2

    if verbose == True:
        print(f"PADDING TYPE: {type(padding)}")
        print(f"PADDING SIZE: {padding.size}")

        print("\nConverting resized img to PIL...")

    img_resized = F.to_pil_image(img_resized)
    if verbose == True:
        print(f"RESIZED IMAGE TYPE: {type(img_resized)}")
        print(f"RESIZED IMAGE SIZE: {img_resized.size}")

        print("\nPasting image to padding...")

    padding.paste(img_resized, (paste_x, paste_y))

    if verbose == True:
        print(f"PASTED IMAGE TYPE {type(padding)}")
        print(f"PASTED IMAGE SIZE {padding.size}")

    scaled_image_tensor = F.to_tensor(padding)

    return scaled_image_tensor


def denormalize(img_t_sqzd, mean=None, std=None, permute=False):
    print(f"\nIMG 2bDENORMED TYPE SIZE:{type(img_t_sqzd)}\n{img_t_sqzd.shape}")
    t_c, t_h, t_w = img_t_sqzd.shape
    denorm_image = torch.empty_like(img_t_sqzd)

    for c in range(t_c):
        for h in range(t_h):
            for w in range(t_w):
                denorm_image[c, h, w] = img_t_sqzd[c, h, w] * std[c] + mean[c]

    if permute == True:
        # denorm_image  = denorm_image.permute(1, 2, 0)
        print(f"IMAGE SIZE TYPE AFTER PERMUTE: {type(denorm_image)}")
        print(f"{denorm_image.shape}")

    denorm_image = transforms.ToPILImage()(denorm_image)
    print(f"DENORM_IMG INFO: {type(denorm_image)}")
    print(f"{denorm_image.size}")

    return denorm_image


def preprocess_img(image_input, device="cpu", plot=False, verbose=False):
    if not isinstance(image_input, Image.Image):
        image_input = Image.open(image_input)

    if plot == True:
        fig, ax = plt.subplots(1, 2)
        ax[0].set_title("Raw Image")
        ax[0].axis("off")
        ax[0].imshow(image_input)

    print("–" * 10)

    if verbose == True:
        print(f"IMAGE TYPE: {type(image_input)}")
        print(f"IMAGE SIZE: {image_input.size}")
        print(f"IMAGE MODE: {image_input.mode}")

    print("\nProcessing image...")

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.Lambda(lambda img: resize_pad(img)),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean, std),
        ]
    )

    image_tensor = transform(image_input).unsqueeze(0).to(device)

    image_tensor = image_tensor.squeeze()

    if verbose == True:
        print(f"\nTRANSFORMED IMAGE TYPE: {type(image_tensor)}")
        print(f"TRANSFORMED IMAGE SIZE: {image_tensor.size()}")
        print(f"SQUEEZED TENSOR SIZE: {image_tensor.size()}")

    denorm_image = denormalize(image_tensor, mean=mean, std=std)

    if plot == True:
        ax[1].set_title("Scaled Image")
        ax[1].axis("off")
        ax[1].imshow(denorm_image)

        try:
            print("Saving scaled image...")
            filename = "./scaled_img.jpeg"
            base, ext = os.path.splitext(filename)
            counter = 1
            while os.path.exists(filename):
                filename = f"{base}_{counter}{ext}"
                counter += 1

            plt.savefig(filename)
            print(f"Saved scaled image as: {filename}")
        except Exception as e:
            print(f"ERROR SAVING scaled image: {e}")

    print("–" * 10)

    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor
