# Built in packages
import math
import sys
import time
from pathlib import Path
import cv2
import numpy as np
# Matplotlib will need to be installed if it isn't already. This is the only package allowed for this base part of the
# assignment.
from matplotlib import pyplot
from matplotlib.patches import Rectangle

# import our basic, light-weight png reader library
import imageIO.png

# this function reads an RGB color png file and returns width, height, as well as pixel arrays for r,g,b
def readRGBImageToSeparatePixelArrays(input_filename):
    img = cv2.imread(input_filename, cv2.IMREAD_COLOR)
    image_height, image_width, _ = img.shape
    print("read image width={}, height={}".format(image_width, image_height))
    pixel_array_b, pixel_array_g, pixel_array_r = cv2.split(img)

    return (image_width, image_height, pixel_array_r, pixel_array_g, pixel_array_b)


# a useful shortcut method to create a list of lists based array representation for an image, initialized with a value
def createInitializedGreyscalePixelArray(image_width, image_height, initValue = 0):

    new_array = [[initValue for x in range(image_width)] for y in range(image_height)]
    return new_array


# You can add your own functions here:
def RGBToGreyscale(img_file):
    image = cv2.imread(img_file, cv2.IMREAD_COLOR)
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    normalized = cv2.normalize(grayscale, None, 0, 255, cv2.NORM_MINMAX)

    return normalized

def StdDev5x5(inputim):
    x = cv2.Sobel(inputim, cv2.CV_64F, 1, 0, ksize=3)
    y = cv2.Sobel(inputim, cv2.CV_64F, 0, 1, ksize=3)

    diff = cv2.absdiff(x, y)

    return diff

def GaussianFiltering(input, repeat):
    img = input.copy()

    for i in range(repeat):
        img = cv2.GaussianBlur(img, (3, 3), 0)

    return img




def ThreshholdImg(inputImg, thresh_val):
    random, img = cv2.threshold(inputImg, thresh_val, 255, cv2.THRESH_BINARY)

    return img


def ErosionSteps5x5(img, repeat):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    output = cv2.erode(img, kernel, iterations=repeat)

    return output

def DilationSteps5x5(img, repeat):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    output = cv2.dilate(img, kernel, iterations=repeat)

    return output


def computeConnectedComponentLabeling(inputImg):
    input_img = np.uint8(inputImg)
    num_labels, lbl_img, stats, centroids = cv2.connectedComponentsWithStats(input_img)

    barcode_bounding_boxes = []
    for i in range(1, num_labels):
        min_x = stats[i, cv2.CC_STAT_LEFT]
        min_y = stats[i, cv2.CC_STAT_TOP]
        max_x = min_x + stats[i, cv2.CC_STAT_WIDTH]
        max_y = min_y + stats[i, cv2.CC_STAT_HEIGHT]

        width = max_x - min_x
        height = max_y - min_y

        # Calculate aspect ratio
        aspect_ratio = max(width, height) / min(width, height)

        # Filter out components with aspect ratio greater than 1.8 for square-like barcodes
        # and aspect ratio greater than 5 for long rectangular barcodes
        if (aspect_ratio <= 1.8 and width * height > 3000) or (aspect_ratio <= 5 and width * height > 3000):
            barcode_bounding_boxes.append((min_x, min_y, max_x, max_y))

    return lbl_img, barcode_bounding_boxes

# This is our code skeleton that performs the barcode detection.
# Feel free to try it on your own images of barcodes, but keep in mind that with our algorithm developed in this assignment,
# we won't detect arbitrary or difficult to detect barcodes!
def main():
    start_time = time.time()
    Extension = Path('Extension')

    command_line_arguments = sys.argv[1:]

    SHOW_DEBUG_FIGURES = True

    # this is the default input image filename
    filename = "Multiple_barcodes"
    input_filename = "images/"+filename+".png"

    if command_line_arguments != []:
        input_filename = command_line_arguments[0]
        SHOW_DEBUG_FIGURES = False

    output_path = Path("output_images")
    if not output_path.exists():
        # create output directory
        output_path.mkdir(parents=True, exist_ok=True)

    output_filename = output_path / Path(filename+"_output.png")
    if len(command_line_arguments) == 2:
        output_filename = Path(command_line_arguments[1])

    # we read in the png file, and receive three pixel arrays for red, green and blue components, respectively
    # each pixel array contains 8 bit integer values between 0 and 255 encoding the color values
    (image_width, image_height, px_array_r, px_array_g, px_array_b) = readRGBImageToSeparatePixelArrays(input_filename)

    # setup the plots for intermediate results in a figure
    fig1, axs1 = pyplot.subplots(2, 2)
    axs1[0, 0].set_title('Input red channel of image')
    axs1[0, 0].imshow(px_array_r, cmap='gray')
    axs1[0, 1].set_title('Input green channel of image')
    axs1[0, 1].imshow(px_array_g, cmap='gray')
    axs1[1, 0].set_title('Input blue channel of image')
    axs1[1, 0].imshow(px_array_b, cmap='gray')

    # STUDENT IMPLEMENTATION here
    print("Converting to grayscale & Normalizing", time.time() - start_time)
    grayImg = RGBToGreyscale(input_filename)
    pyplot.imsave(Extension / 'RGB2Grayscale.png', grayImg, cmap='gray')
    print("Applying Standard Deviation Filter", time.time() - start_time)
    sobel = StdDev5x5(grayImg)
    pyplot.imsave(Extension / 'stdev.png', sobel, cmap='gray')
    print("Gaussian Filtering", time.time() - start_time)
    gFiltered = GaussianFiltering(sobel, 4)
    pyplot.imsave(Extension / 'Gaussian.png', gFiltered, cmap='gray')
    print("Applying Threshold", time.time() - start_time)
    threshImg = ThreshholdImg(gFiltered, 100)
    pyplot.imsave(Extension / 'Threshold.png', threshImg, cmap='gray')
    print("Applying Erosion", time.time() - start_time)
    erodedImg = ErosionSteps5x5(threshImg, 4)
    pyplot.imsave(Extension / 'Erosion.png', erodedImg, cmap='gray')
    print("Applying Dilation", time.time() - start_time)
    dilatedImg = DilationSteps5x5(erodedImg, 5)
    pyplot.imsave(Extension / 'Dilation.png', dilatedImg, cmap='gray')
    print("Finding barcodes", time.time() - start_time)
    px_array, bounds = computeConnectedComponentLabeling(dilatedImg)


    def seperateArraysToRGB(px_array_r, px_array_g, px_array_b, image_width, image_height):
        new_array = [[[0 for c in range(3)] for x in range(image_width)] for y in range(image_height)]

        for y in range(image_height):
            for x in range(image_width):
                new_array[y][x][0] = px_array_r[y][x]
                new_array[y][x][1] = px_array_g[y][x]
                new_array[y][x][2] = px_array_b[y][x]

        return new_array

    px_array = seperateArraysToRGB(px_array_r, px_array_g, px_array_b, image_width, image_height)








    # Compute a dummy bounding box centered in the middle of the input image, and with as size of half of width and height
    # Change these values based on the detected barcode region from your algorithm


    for coords in bounds:
        min_x, min_y, max_x, max_y = coords
        bbox_min_x = min_x
        bbox_max_x = max_x
        bbox_min_y = min_y
        bbox_max_y = max_y

    # The following code is used to plot the bounding box and generate an output for marking
    # Draw a bounding box as a rectangle into the input image
        axs1[1, 1].set_title('Final image of detection')
        axs1[1, 1].imshow(px_array, cmap='gray')
        rect = Rectangle((bbox_min_x, bbox_min_y), bbox_max_x - bbox_min_x, bbox_max_y - bbox_min_y, linewidth=1,
                     edgecolor='purple', facecolor='none')
        axs1[1, 1].add_patch(rect)
    print("Finished", time.time() - start_time)

    # write the output image into output_filename, using the matplotlib savefig method
    extent = axs1[1, 1].get_window_extent().transformed(fig1.dpi_scale_trans.inverted())
    pyplot.savefig(output_filename, bbox_inches=extent, dpi=600)

    if SHOW_DEBUG_FIGURES:
        # plot the current figure
        pyplot.show()


if __name__ == "__main__":
    main()