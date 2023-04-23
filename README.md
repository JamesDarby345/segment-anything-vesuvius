# Segment Anything? Segment A Vesuvius Scroll!

A data processing pipeline using meta's Segment Anything model to segment the Vesuvius Scroll .tif files and mask the superfluous data with black pixels allowing for lossless compression (LZW) reducing .tif file size by ~70% (120MB -> ~40MB) while preserving all the useful data.

## TLDR Overview


## Setup
This repo isnt particularly friendly if you want to setup the pipeline yourself as youll need all of dependencies for https://github.com/facebookresearch/segment-anything, along with the folder structure and file names I used to organize the Vesuvius Scroll .tif files. It would be possible to setup with a bit of fiddling around with file path variables though. I may in the future make this easier, but the goal of reducing the .tif file sizes while maintaing all useful information only requires the data to be processed once. Though this technique may be useful for other, unrelated applications.

## Results
I make the claim that this process can reduce the .tif file size by ~70% while maintaining all useful information. 
The below images show the easy part to demonstrate, the reduced file size of the first file, 00000.tif (121MB) -> c00000.tif (31.4MB), while maintining the exact same image dimensions and file type.<br>
![image](https://user-images.githubusercontent.com/49734270/233865658-4b3342cc-fc3c-48f0-97aa-e9e51ae53a76.png)

![image](https://user-images.githubusercontent.com/49734270/233865689-c30718b6-ae75-41be-9e2a-1a453c07b030.png)

I have made the assumption that, the surrounding area [green], the case [yellow] and the detached 'wrap' [red] do not contain useful information for reading the scrolls.

![image](https://user-images.githubusercontent.com/49734270/233868534-652f526c-dd2c-4ef1-b884-c7c333fd544f.png)

Using that assumption I segmented the image, and set every pixel not part of the scroll to black (0,0,0).

**Original Image**
![image](https://user-images.githubusercontent.com/49734270/233868711-593be44b-ced3-42f0-973a-b84f923fc552.png)

**Result of Image processing Pipeline**
![image](https://user-images.githubusercontent.com/49734270/233868736-9e6b8afb-1917-47dd-a64c-d7f638d0e5bb.png)
*Note these images are not the originals, but smaller files that are under githubs upload threshold. Also note the mask over the scroll has been enlarged by 0.5% to help ensure any pixels containing papyri are not covered, this amount can be enlarged if adversarial examples are produced.

To convince myself that the lossless compression was in fact lossless I carried out a series of experiments to ensure none of the pixels in the scroll had been changed. I did this with the computeImageDiff files. The basic idea is to compare the original image, and the new compressed image by taking the absolute difference of each of their pixels and producing an image with that value as the value of each pixel. This should produce an image where the scroll portionis completly black while the background is the original background as black contributes 0. That verification technique produced the following image:

![image](https://user-images.githubusercontent.com/49734270/233869107-b73416a7-c424-41e4-9dec-02e90fb360d5.png)

In addition I produced a version that multiplied the difference by 100, so even a difference in value of 1 would be noticeable. This does mean the background becomes noisy, but the uninterupted black shape of the scroll is what it important. This version produced the following image: 

![image](https://user-images.githubusercontent.com/49734270/233869242-afe3abb0-16da-4136-bbcf-125182ed5064.png)

I also produced a version that changed the color of any difference in pixels to red (computeImageDiffColor,py), and an updated version that added a radius so even a single pixel changed by a single value would become obvious (computeImageDiffColorRadius.py). The radius version produced the following result: 

![image](https://user-images.githubusercontent.com/49734270/233869380-e4623615-c556-493c-b148-85c32a07d7c1.png)

To verify the scripts did what I thought they did, I edited 06052.tif by drawing on it with a 1 pixel wide brush to a grey that closely matched the scroll.
Here is the 'corrupted' image:
![image](https://user-images.githubusercontent.com/49734270/233869486-e4eda257-0fc0-467f-8536-51fb3f121ca7.png)

And the result of the color radius script:
![image](https://user-images.githubusercontent.com/49734270/233869501-445dbc7b-3fdb-4ff7-9064-32322e1edd21.png)

As you can see it clearly brought forth any changes in the images pixel values. Thus I am convinced that this technique does in fact not change the pixel values of the images where the papri and ink are located, but does reduce the file size by nearly 70%. Please feel free to validate these findings. 

## Pipeline process
The pipeline carries out a number of steps to produce these results which I will go over on a high level in this section. The main file scrollSegRLESeqRange.py takes a range of images specified by their number and processes them one at a time. The first step is to make a downsized version of the original image, this is so the image (batch) can fit it the systems VRAM in one go during the segmenting process. I found 10x downsizing fit in my 8GB of VRAM on my GPU. This smaller image is used to produce the masks with meta's Segment Anything model. The pipeline then uses the dumb heuristic that the first (largest) mask is the scroll. This has some shortcomings as explained in the next section. Next the downsized mask is then scaled back up to the size of the original image, and reshaped to exactly match the original images dimensions. Next the mask is enlarged by 0.5% to ensure none of the scroll (useful information) is covered by the mask as a result of the scaling process. Note that the original image is not scaled in this process, only the mask is upscaled. Then a new image is created that sets all pixels not covered by the mask to black and uses the pixel values of the original image where the mask does cover it. This image is then saved with LZW compression, and the large area of uniform black pixels allows the compression to drastically reduce the .tif file size. As seen in the results section, the original pixels in the scroll maintain their original value.

## Limitations & potential improvements
Due to the simplistic mask selection of choosing the largest mask, some development work or manual intervention would be needed for cases where the scroll does not constitue the largest single section/mask of the image. This started to happen somewhere between file 12600.tif and 13000.tif on the first scroll. But the method does mean it will work as is for the majority of the scroll. This also means all the data should be quickly manually screened via visual inspection to make sure the correct mask was choosen. This can easily be done by viewing the image thumbnails.

Example of model starting to choose incorrect masks:<br>
![image](https://user-images.githubusercontent.com/49734270/233870282-2c1b7101-40e8-47b8-9061-54ad8bf53de3.png)











