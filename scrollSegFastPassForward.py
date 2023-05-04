import time
from tifffile import tifffile
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
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
from helpers import *

# from scipy.ndimage import center_of_mass

Image.MAX_IMAGE_PIXELS = 150_000_000

print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
cuda_available = torch.cuda.is_available()
print("CUDA is available:", cuda_available)
torch.cuda.empty_cache()

# print("sleeping...")
# time.sleep(3.5 * 3600)
# print("finished sleeping")
# sam_checkpoint = "segment-anything\sam_vit_l_0b3195.pth"
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"
# filePath = "../../Scroll2/fullScroll2Data/"
filePath = "../../fullScrollData/"


sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
if cuda_available:
    sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(
    model=sam, output_mode="coco_rle", min_mask_region_area=0
)
predictor = SamPredictor(sam)

scale_factor = 0.1  # Reduce the dimensions of the original size so batch can fit in GPU memory (8GB)
dilation = 2  # Dilate the mask by 3% to provide a buffer so that the entire scroll region is covered

formatted_numbers = []
# formatted_numbers += [f"{n:05d}" for n in range(14353, 14375 + 1)]
formatted_numbers += [f"{n:05d}" for n in range(13076, 13076 + 1)]
formatted_numbers += [f"{n:05d}" for n in range(13200, 13200 + 1)]
formatted_numbers += [f"{n:05d}" for n in range(13400, 13400 + 1)]
formatted_numbers += [f"{n:05d}" for n in range(13600, 13600 + 1)]
formatted_numbers += [f"{n:05d}" for n in range(13800, 13800 + 1)]
formatted_numbers += [f"{n:05d}" for n in range(14000, 14000 + 1)]
formatted_numbers += [f"{n:05d}" for n in range(14353, 14353 + 1)]

chosen_mask = 0
# output_folder = "Scroll2/processedData00000-01000"
output_folder = "fullScrollDataTest"
mask_freq = 1
manuallyChooseMask = 1

previous_mask = np.array([0])

print(formatted_numbers[0], formatted_numbers[-1], len(formatted_numbers))

startTime = time.time()
i = 0
for number in formatted_numbers:
    try:
        timeStartRead = time.time()
        image = read_image_tifffile(filePath + number + ".tif")
        print(number + " read time:", time.time() - timeStartRead)
    except:
        record_skipped_image(number)
        continue
    if image is None:
        record_skipped_image(number)
        continue
    if i % mask_freq == 0:
        imageToMask = cv2.imread(filePath + number + ".tif")
        downsampled_image = downsample_image(imageToMask, scale_factor)
        torch.cuda.empty_cache()
        print(filePath + number + ".tif")
        predictor.set_image(downsampled_image)
        masks = None
        if previous_mask.all() == 0:
            input_points = np.array(
                [
                    [downsampled_image.shape[0] / 2, downsampled_image.shape[1] / 2],
                    [0, 0],
                    # [downsampled_image.shape[0], downsampled_image.shape[1]],
                ],
            )
            input_labels = np.array([1, 0])
            startTimeM = time.time()
            mask, _, _ = predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
            )
            print(number + " mask generation time:", time.time() - startTimeM)
        else:
            startTimeM = time.time()
            mask = predictor.predict(
                mask_input=previous_mask,
            )
            print(number + " mask predict generation time:", time.time() - startTimeM)
        print(mask.shape)

        # if manuallyChooseMask:
        #     chosen_mask = manually_choose_mask(downsampled_image, masks)

        # scaled_rle_mask = scale_rle_mask(
        #     masks[chosen_mask]["segmentation"], 1 / scale_factor, image.shape
        # )
        scaled_mask = scale_mask(mask, 1 / scale_factor, image.shape)

    # rle_image = visualize_rle_mask(image, scaled_rle_mask)
    # show_image(rle_image)

    # applied_mask = apply_mask(image, scaled_rle_mask)
    # show_image(applied_mask)
    previous_mask = mask
    dilated_applied_mask = apply_dilated_mask(image, scaled_mask, dilation, True)
    #  print(dilated_applied_mask.dtype)
    # show_image(dilated_applied_mask)

    # Save the masked image as a 16-bit grayscale TIFF file
    outputFilePath = "../../" + output_folder + "/" + number + ".tif"
    outputCheck = os.path.dirname(outputFilePath)
    if not os.path.exists(outputCheck):
        raise ValueError("Output directory does not exist")
    startTimeW = time.time()
    written = write_image_cv2(outputFilePath, dilated_applied_mask)
    print(number + " write time:", time.time() - startTimeW)

    if not written:
        print("Failed to write image to", outputFilePath)

    if i % mask_freq == 0 or i % mask_freq == mask_freq - 1:
        outputFilePathVerification = (
            "../../compressedScrollDataVerification/" + number + ".tif"
        )
        cv2.imwrite(
            outputFilePathVerification,
            dilated_applied_mask,
        )
    i += 1
print("Time taken:", time.time() - startTime, " seconds")
