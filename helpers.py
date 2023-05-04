import time
from tifffile import tifffile
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from pycocotools import mask as coco_mask
import numpy as np
import cv2
from PIL import Image
import torch
import torchvision
import matplotlib.pyplot as plt
import os
from scipy import ndimage
import tkinter as tk
from tkinter import simpledialog


def read_image_tifffile(file_path):
    return tifffile.imread(file_path)
    # return cv2.imread(filePath + number + ".tif", cv2.IMREAD_ANYDEPTH)


def write_image_cv2(file_path, image):
    try:
        cv2.imwrite(
            file_path,
            image,
        )
    except Exception as e:
        print(e)
        return False
    return True


def write_image_tifffile(file_path, image):
    try:
        tifffile.imwrite(file_path, image, compression="lzw")
    except Exception as e:
        print(e)
        return False
    return True


def get_number():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    num = simpledialog.askinteger(
        "Enter a number", "Please enter a number:"
    )  # Show the popup
    root.destroy()  # Destroy the main window when done
    return num


# coco_rle visualization
def visualize_rle_mask(image, rle_mask, alpha=0.5):
    # Decode the RLE mask into a binary mask
    binary_mask = coco_mask.decode(rle_mask)

    # Overlay the binary mask on the image
    masked_image = image.copy()
    masked_image[binary_mask == 1] = (
        masked_image[binary_mask == 1] * (1 - alpha) + np.array([255, 0, 0]) * alpha
    ).astype(np.uint8)

    return masked_image


def show_image(image):
    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    plt.axis("on")
    plt.show()


def remove_disconnected_regions(binary_mask, predict_model_used=False):
    if not predict_model_used:
        labeled_mask, _ = ndimage.label(binary_mask)
        height, width = binary_mask.shape
    else:
        height, width = binary_mask[-2:]
        mask_to_label = binary_mask.reshape(height, width, 1)
        labeled_mask, _ = ndimage.label(mask_to_label)
    center_y, center_x = height // 2, width // 2
    central_label = labeled_mask[center_y, center_x]
    central_region_mask = labeled_mask == central_label
    return central_region_mask


def scale_rle_mask(rle_mask, scale_factor, image_size=(1024, 1024)):
    # Decode the RLE mask into a binary mask
    binary_mask = coco_mask.decode(rle_mask)
    binary_mask = remove_disconnected_regions(binary_mask)

    # Resize the binary mask
    new_height = int(binary_mask.shape[0] * scale_factor)
    new_width = int(binary_mask.shape[1] * scale_factor)
    resized_binary_mask = cv2.resize(
        binary_mask.astype(np.uint8),
        (new_width, new_height),
        interpolation=cv2.INTER_NEAREST,
    )

    resized_binary_mask = cv2.resize(
        binary_mask.astype(np.uint8), (image_size[1], image_size[0])
    )

    # Encode the resized binary mask back into an RLE mask
    scaled_rle_mask = coco_mask.encode(np.asfortranarray(resized_binary_mask))
    return scaled_rle_mask


def scale_mask(binary_mask, scale_factor, image_size=(1024, 1024)):
    binary_mask = remove_disconnected_regions(binary_mask, True)

    new_height = int(binary_mask.shape[0] * scale_factor)
    new_width = int(binary_mask.shape[1] * scale_factor)
    resized_binary_mask = cv2.resize(
        binary_mask.astype(np.uint8),
        (new_width, new_height),
        interpolation=cv2.INTER_NEAREST,
    )

    resized_binary_mask = cv2.resize(
        binary_mask.astype(np.uint8), (image_size[1], image_size[0])
    )

    return resized_binary_mask


def downsample_image(image, scale_factor):
    new_width = int(image.shape[1] * scale_factor)
    new_height = int(image.shape[0] * scale_factor)
    new_size = (new_width, new_height)

    # Use cv2.resize() to downsample the image
    resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return resized_image


# fast
def apply_dilated_mask(image, mask, dilation_percentage, is_bin_mask=False):
    # Decode the RLE mask into a binary mask
    if not is_bin_mask:
        binary_mask = coco_mask.decode(mask)
    else:
        binary_mask = mask

    # Dilate the binary mask
    kernel_size = int(max(binary_mask.shape) * dilation_percentage / 100)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)

    # Multiply the original image by the dilated binary mask
    if len(image.shape) == 3:
        masked_image = image * np.stack([dilated_mask] * 3, axis=-1)
    else:
        masked_image = image * dilated_mask
    return masked_image


def record_skipped_image(number):
    with open("../../compressedScrollDataVerification/missing.txt", "a") as myfile:
        myfile.write(number + "\n")


def manually_choose_mask(downsampled_image, masks):
    rle_image = visualize_rle_mask(downsampled_image, masks[0]["segmentation"])
    show_image(rle_image)
    rle_image = visualize_rle_mask(downsampled_image, masks[1]["segmentation"])
    show_image(rle_image)
    rle_image = visualize_rle_mask(downsampled_image, masks[2]["segmentation"])
    show_image(rle_image)
    rle_image = visualize_rle_mask(downsampled_image, masks[3]["segmentation"])
    show_image(rle_image)
    return get_number()
