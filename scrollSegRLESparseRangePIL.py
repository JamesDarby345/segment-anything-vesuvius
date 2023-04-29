import time
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from pycocotools import mask as coco_mask
import numpy as np
import cv2
from PIL import Image
import torch
import torchvision
import matplotlib.pyplot as plt
import os

Image.MAX_IMAGE_PIXELS = 150_000_000

print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
cuda_available = torch.cuda.is_available()
print("CUDA is available:", cuda_available)
torch.cuda.empty_cache()

# print("sleeping...")
# time.sleep(4 * 3600)
# print("finished sleeping")
# sam_checkpoint = "segment-anything\sam_vit_l_0b3195.pth"
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"
# filePath = "../../Scroll2/fullScroll2Data/"
filePath = "../../fullScrollData/"


def read_image_pil(file_path):
    with Image.open(file_path) as img:
        return np.array(img)


def write_image_pil(file_path, image):
    try:
        img = Image.fromarray(image)
        img = img.convert("I;16")
        img.save(file_path, compression="tiff_lzw")
    except:
        return False
    return True


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
    plt.axis("off")
    plt.show()


def scale_rle_mask(rle_mask, scale_factor, image_size=(1024, 1024)):
    # Decode the RLE mask into a binary mask
    binary_mask = coco_mask.decode(rle_mask)

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


def downsample_image(image, scale_factor):
    new_width = int(image.shape[1] * scale_factor)
    new_height = int(image.shape[0] * scale_factor)
    new_size = (new_width, new_height)

    # Use cv2.resize() to downsample the image
    resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return resized_image


# fast
def apply_dilated_mask(image, rle_mask, dilation_percentage):
    # Decode the RLE mask into a binary mask
    binary_mask = coco_mask.decode(rle_mask)

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


sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
if cuda_available:
    sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(
    model=sam, output_mode="coco_rle", min_mask_region_area=0
)  # ,output_mode="coco_rle

scale_factor = 0.1  # Reduce the dimensions of the original size so batch can fit in GPU memory (8GB)
dilation = 2  # Dilate the mask by 3% to provide a buffer so that the entire scroll region is covered

formatted_numbers = []
formatted_numbers += [
    f"{n:05d}" for n in range(0, 1)  # end of range needs to be 1 higher than you want
]  # change the range to change which images are processed

chosen_mask = 0
# output_folder = "Scroll2/processedData00000-01000"
output_folder = "processedData00000-02000"
mask_freq = 50

# formatted_numbers += [
#     f"{n:06d}" for n in range(12500, 12505)
# ]  # artifact of how the images are named when I downloaded them, adding a 0 in front of all the 5 digit numbers

print(formatted_numbers[0], formatted_numbers[-1], len(formatted_numbers))

startTime = time.time()
i = 0
for number in formatted_numbers:
    try:
        timeStartRead = time.time()
        image = read_image_pil(filePath + number + ".tif")
        print(number + " read time:", time.time() - timeStartRead)
    except:
        record_skipped_image(number)
        continue
    if image is None:
        record_skipped_image(number)
        continue
    if i % mask_freq == 0:
        imageToMask = cv2.imread(filePath + number + ".tif")
        torch.cuda.empty_cache()
        print(filePath + number + ".tif")

        startTimeM = time.time()
        downsampled_image = downsample_image(imageToMask, scale_factor)
        masks = mask_generator.generate(downsampled_image)
        print(number + " mask generation time:", time.time() - startTimeM)

        # print(len(masks))
        # print(masks[0].keys())
        # ind = 0
        # for m in masks:
        #     print(m["area"], ind)
        #     ind += 1

        # rle_image = visualize_rle_mask(downsampled_image, masks[0]["segmentation"])
        # show_image(rle_image)

        scaled_rle_mask = scale_rle_mask(
            masks[chosen_mask]["segmentation"], 1 / scale_factor, image.shape
        )
    # rle_image = visualize_rle_mask(image, scaled_rle_mask)
    # show_image(rle_image)

    # applied_mask = apply_mask(image, scaled_rle_mask)
    # show_image(applied_mask)

    dilated_applied_mask = apply_dilated_mask(image, scaled_rle_mask, dilation)
    # show_image(dilated_applied_mask)

    # Save the masked image as a 16-bit grayscale TIFF file
    outputFilePath = "../../" + output_folder + "/" + number + ".tif"
    outputCheck = os.path.dirname(outputFilePath)
    if not os.path.exists(outputCheck):
        raise ValueError("Output directory does not exist")
    startTimeW = time.time()
    written = write_image_pil(outputFilePath, dilated_applied_mask)
    print(number + " write time:", time.time() - startTimeW)

    if not written:
        print("Failed to write image to", outputFilePath)

    if i % mask_freq == 0 or i % mask_freq == mask_freq - 1:
        outputFilePathVerification = (
            "../../compressedScrollDataVerification/" + number + ".tif"
        )
        write_image_pil(outputFilePath, dilated_applied_mask)
    i += 1
print("Time taken:", time.time() - startTime, " seconds")
