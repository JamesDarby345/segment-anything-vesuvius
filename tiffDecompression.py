import os
from tifffile import imread, imwrite
import sys

def convert_to_uncompressed(input_folder, output_folder, delete_original=False):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    tiff_files = [f for f in os.listdir(input_folder) if f.endswith('.tif') or f.endswith('.tiff')]

    for tiff_file in tiff_files:
        input_path = os.path.join(input_folder, tiff_file)
        output_path = os.path.join(output_folder, tiff_file)

        image = imread(input_path)
        
        # Save as an uncompressed 16-bit grayscale image
        imwrite(output_path, image, compress=0)
        
        # Optionally delete the original file
        if delete_original:
            os.remove(input_path)

input_folder, output_folder = sys.argv[1], sys.argv[2]
delete_original = sys.argv[3].lower() == 'true' if len(sys.argv) > 3 else False

convert_to_uncompressed(input_folder, output_folder, delete_original)

#usage: python tiffDecompression.py <input_folder> <output_folder> [delete_original]
#default is to not delete the original files
