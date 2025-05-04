import base64
import json
import os

import matplotlib
import matplotlib.pyplot as plt
import requests
import torch
from PIL import Image
from torchvision import transforms

from preprocess import denormalize, preprocess_img

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def conv_img_bs64(img_path):
    print("–" * 10)
    print("CONVERTING to BS64 STRING...")
    with open(img_path, "rb") as img:
        bs64_string = base64.b64encode(img.read()).decode("utf-8")
    # print(f'SHOWING bs64 string:\n{bs64_string}\n')
    return bs64_string


def json_load(img_path, img_bs64_string):
    print("–" * 10)
    print("WRITING PAYLOAD...")
    payload = {
        "name": img_path.split("/")[-1],
        "image": f"data:image/jpeg;base64, {img_bs64_string}",
    }
    # print(f'SHOWING PAYLOAD...\n{payload}\n')
    return json.dumps(payload)


def send_img(img_path, url, verbose=False):
    img_bs64_string = conv_img_bs64(img_path)
    request = json_load(img_path, img_bs64_string)

    print("–" * 10)
    print("CREATING HEADERS...")
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, data=request, headers=headers)

    if response.status_code == 200:
        print("–" * 10)
        print("IMAGE PROCESSED")
        if verbose == True:
            print(f"RESPONSE: {response.json()}")
        return response.json()
    else:
        print("–" * 10)
        print("IMAGE PROCESSING FAILED")


def process_response(result, image_input, verbose=False, conv_tensor=False):

    image = Image.open(image_input)
    num_animals = len(result)
    cmap = matplotlib.colormaps["rainbow"]

    fig, ax = plt.subplots(1, 2)

    for i, animal_data in enumerate(result):
        try:
            animal_data.items()
        except Exception as e:
            import sys

            sys.exit(f"\nWHERE FACE??? {e}")

        if verbose == True:
            print(f"\n[{i}] ANIMAL DATA:\n {animal_data}")
            print(f"\nANIMAL_DATA.ITEMS():\n {animal_data.items()}")

        color = cmap(i / num_animals + 2)

        for animal, details in animal_data.items():

            bbox = details["bbox"]
            pul = bbox.get("pul")
            pbr = bbox.get("pbr")

            if verbose == True:
                print(f"\nShowing ANIMAL: {animal}\nDETAILS: {details}")

            landmarks = details["landmarks"]
            for landmark in landmarks:
                x = landmark["x"]
                y = landmark["y"]
                ax[0].scatter(x, y, color=color, s=10)

        ax[0].set_title("Original")
        ax[0].axis("off")
        ax[0].imshow(image)

        if pul and pbr:
            bbox_x1 = pul.get("x")
            bbox_y1 = pul.get("y")
            bbox_x2 = pbr.get("x")
            bbox_y2 = pbr.get("y")

            if all(
                [
                    isinstance(c, (int, float))
                    for c in [bbox_x1, bbox_y1, bbox_x2, bbox_y2]
                ]
            ):
                bbox_x1, bbox_y1, bbox_x2, bbox_y2 = (
                    int(bbox_x1),
                    int(bbox_y1),
                    int(bbox_x2),
                    int(bbox_y2),
                )

                bbox_w = bbox_x2 - bbox_x1
                bbox_h = bbox_y2 - bbox_y1

                print(f"\nProcessing BBOX for data [{i}]...")

                rect = matplotlib.patches.Rectangle(
                    (bbox_x1, bbox_y1),
                    bbox_w,
                    bbox_h,
                    linewidth=2,
                    edgecolor=color,
                    facecolor="none",
                    label=f"{i} DATA BBOX",
                )
                ax[0].add_patch(rect)

                bbox_w = bbox_x2 - bbox_x1
                bbox_h = bbox_y2 - bbox_y1

                crop_box = (bbox_x1, bbox_y1, bbox_x2, bbox_y2)

                try:
                    print("Cropping image...")
                    cropped_image = image.crop(crop_box)

                    cropped_w, cropped_h = cropped_image.size
                    print(f"CROPPED DIMS:{cropped_w}x{cropped_h}")

                    print("Scaling image...")
                    cropped_image = preprocess_img(
                        cropped_image, device=device, plot=False, verbose=True
                    )
                    print(f"CROPPED IMG TYPE: {type(cropped_image)}")

                    mean = [0.485, 0.456, 0.406]
                    std = [0.229, 0.224, 0.225]

                    cropped_image = cropped_image.squeeze(0)
                    print(f"IMAGE TYPE SIZE BEFORE DENORM:{type(cropped_image)}")
                    print(f"{cropped_image.shape}")

                    target_size = (224, 224)
                    target_h, target_w = target_size

                    aspect_ratio = cropped_w / cropped_h

                    if cropped_w > cropped_h:
                        new_w = target_w
                        new_h = int(target_w / aspect_ratio)
                    else:
                        new_h = target_h
                        new_w = int(target_h * aspect_ratio)

                    paste_x = (target_w - new_w) // 2
                    paste_y = (target_h - new_h) // 2

                    cropped_image = denormalize(
                        cropped_image, mean=mean, std=std, permute=True
                    )
                    # cropped_image = cropped_image.numpy()
                    ax[1].axis("off")
                    ax[1].set_title("Cropped")
                    ax[1].imshow(cropped_image)

                    for animal, details in animal_data.items():

                        landmarks = details["landmarks"]
                        for landmark in landmarks:
                            x = landmark["x"]
                            y = landmark["y"]

                            x_r = x - bbox_x1
                            y_r = y - bbox_y1

                            x_sc = x_r * (new_w / cropped_w)
                            y_sc = y_r * (new_h / cropped_h)

                            x_fn = x_sc + paste_x
                            y_fn = y_sc + paste_y

                            ax[1].scatter(x_fn, y_fn, color=color, s=10)

                    try:
                        print("Saving processed image...")
                        filename = "./processed_image.jpeg"
                        base, ext = os.path.splitext(filename)
                        counter = 1
                        while os.path.exists(filename):
                            filename = f"{base}_{counter}{ext}"
                            counter += 1

                        plt.savefig(filename)
                        print(f"Saved processed image as: {filename}")
                        if conv_tensor == True:
                            print("CONVERTIN TO TENSOR...")
                            cropped_image = transforms.ToTensor()(cropped_image)

                            print(
                                f"CONVERTED TYPE SIZE:{type(cropped_image)}\n{cropped_image.shape}"
                            )
                            cropped_image = cropped_image.unsqueeze(0)
                            print(f"SHAPE AFTER UNSQUEEZE: {cropped_image.shape}")
                            return cropped_image
                        else:
                            return cropped_image
                    except Exception as e:
                        print(f"ERROR SAVING processed image: {e}")

                except Exception as e:
                    print(f"Error cropping... ERROR {e}")
            else:
                print("BBOX NOT VALID NUMBERS")
        else:
            print("BBOX MISSING 'pul' or 'pbr'")
