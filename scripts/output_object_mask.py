from PIL import Image
import os
import sys
import numpy as np


def output_object_mask(root_folder, input_folder, output_folder):
    current_folder = os.path.join(root_folder, input_folder)
    print(current_folder)
    images = [f for f in os.listdir(current_folder) if f.endswith('.png')]
    mask_name = input_folder.split("_")[0]
    img_array = np.array(Image.open(os.path.join(current_folder, images[0])).convert("L"))
    width, height = img_array.shape
    print(mask_name, width, height)
    mask = np.zeros((width, height), dtype=np.uint8)
    for img in images:
        img_array = np.array(Image.open(os.path.join(current_folder, img)).convert("L"))
        both_zero_mask = (img_array == 0) & (mask == 0)
        mask = np.where(both_zero_mask, 0, 255).astype(np.uint8)
    mask_image = Image.fromarray(mask)
    output_full_path = os.path.join(root_folder, output_folder)
    os.makedirs(output_full_path, exist_ok=True)
    mask_image.save(os.path.join(output_full_path, f"{mask_name}_mask.png"))
    return

root_folder = "/home/mass_spec_img/mass_spec_img/data/raw_img_v2/"
input_folders = ["SQ1631_images", "SQ1632_images", "SQ1633_images", "SQ1634_images", "SQ1635_images", "SQ1636_images"]
output_folders = ["SQ1631_mask", "SQ1632_mask", "SQ1633_mask", "SQ1634_mask", "SQ1635_mask", "SQ1636_mask"]
for i in range(len(input_folders)):
    output_object_mask(root_folder, input_folders[i], output_folders[i])