# Built in packages
import math
import sys
from pathlib import Path

# Matplotlib will need to be installed if it isn't already. This is the only package allowed for this base part of the 
# assignment.
from matplotlib import pyplot
from matplotlib.patches import Rectangle

# import our basic, light-weight png reader library
import imageIO.png

# this function reads an RGB color png file and returns width, height, as well as pixel arrays for r,g,b
def readRGBImageToSeparatePixelArrays(input_filename):

    image_reader = imageIO.png.Reader(filename=input_filename)
    # png reader gives us width and height, as well as RGB data in image_rows (a list of rows of RGB triplets)
    (image_width, image_height, rgb_image_rows, rgb_image_info) = image_reader.read()

    print("read image width={}, height={}".format(image_width, image_height))

    # our pixel arrays are lists of lists, where each inner list stores one row of greyscale pixels
    pixel_array_r = []
    pixel_array_g = []
    pixel_array_b = []

    for row in rgb_image_rows:
        pixel_row_r = []
        pixel_row_g = []
        pixel_row_b = []
        r = 0
        g = 0
        b = 0
        for elem in range(len(row)):
            # RGB triplets are stored consecutively in image_rows
            if elem % 3 == 0:
                r = row[elem]
            elif elem % 3 == 1:
                g = row[elem]
            else:
                b = row[elem]
                pixel_row_r.append(r)
                pixel_row_g.append(g)
                pixel_row_b.append(b)

        pixel_array_r.append(pixel_row_r)
        pixel_array_g.append(pixel_row_g)
        pixel_array_b.append(pixel_row_b)

    return (image_width, image_height, pixel_array_r, pixel_array_g, pixel_array_b)


# a useful shortcut method to create a list of lists based array representation for an image, initialized with a value
def createInitializedGreyscalePixelArray(image_width, image_height, initValue = 0):

    new_array = [[initValue for x in range(image_width)] for y in range(image_height)]
    return new_array


# You can add your own functions here:
def RGBToGreyscale(image_width, image_height, pixel_array_r, pixel_array_g, pixel_array_b):
    greyscale_pixel_array = createInitializedGreyscalePixelArray(image_width, image_height)
    for i in range(image_height):
        for j in range(image_width):
            greyscale_pixel_array[i][j] = round(0.299*pixel_array_r[i][j] + 0.587*pixel_array_g[i][j] + 0.114*pixel_array_b[i][j])

    return greyscale_pixel_array

def ScaleAndQuantize(pixel_arr, image_width, image_height):
    minPixel = min([min(image_width) for image_width in pixel_arr])
    maxPixel = max([max(image_width) for image_width in pixel_arr])

    for i in range(image_height):
        for j in range(image_width):
            if (pixel_arr[i][j] - minPixel == 0):
                pixel_arr[i][j] = 0.0
            else:
                pixel_arr[i][j] = round((pixel_arr[i][j] - minPixel)*((255.0)/(maxPixel - minPixel)))
    return pixel_arr

def StdDev5x5(pixel_array, image_width, image_height):
    arr = createInitializedGreyscalePixelArray(image_width, image_height, 0)

    for i in range(2, image_height - 2):
        for j in range(2, image_width - 2):
            sum_pixels = 0
            squared_pixels = 0

            for k in range(-2, 3):
                for l in range(-2, 3):
                    sum_pixels += pixel_array[i+k][j+l]

            avg = sum_pixels/25.0
            for k in range(-2, 3):
                for l in range(-2, 3):
                    currVal = pixel_array[i+k][j+l]
                    squared_pixels += pow(currVal-avg, 2)

            arr[i][j] = math.sqrt(squared_pixels/25.0)

    return arr

def Gaussian3x3(pixel_array, image_width, image_height):
    arr = createInitializedGreyscalePixelArray(image_width, image_height)
    g = [[1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]]

    for i in range(image_height):
        for j in range(image_width):
            sum = 0
            for k in range(-1, 2):
                for l in range(-1, 2):
                    r = min(image_height-1, max(i+k, 0))
                    c = min(image_width-1, max(j+l, 0))
                    sum += g[k+1][l+1] * pixel_array[r][c]
            arr[i][j]= sum/16.0
    return arr


def ThreshholdImg(pixel_array, threshhold_val, image_width, image_height):
    for i in range(image_height):
        for j in range(image_width):
            pixel_array[i][j] = 0.0 if pixel_array[i][j] < threshhold_val else 255.0

    return pixel_array


def ErosionSteps5x5(pixel_array, image_width, image_height):
    arr = createInitializedGreyscalePixelArray(image_width, image_height)

    for i in range(2, image_height - 2):
        for j in range(2, image_width - 2):
            flag = True
            for k in range(-2, 3):
                for l in range(-2, 3):
                    if pixel_array[i + k][j + l] <= 0:
                        flag = False

            if flag:
                arr[i][j] = 1

    return arr

def DilationSteps5x5(pixel_array, image_width, image_height):
    img = createInitializedGreyscalePixelArray(image_width, image_height)

    for i in range(image_height):
        for j in range(image_width):
            for k in range(-2, 3):
                for l in range(-2, 3):
                    if 0 <= i + k < image_height and 0 <= j + l < image_width:
                        if pixel_array[i + k][j + l] > 0:
                            img[i][j] = 1

    return img


class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0, item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)

def computeConnectedComponentLabeling(pixel_array, image_width, image_height):
    q = Queue()
    lbl_img = createInitializedGreyscalePixelArray(image_width, image_height)
    lblid = 1
    lblcnt = {}
    sides = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    minMaxVals = [[] for _ in range(100)]

    for i in range(image_height):
        for j in range(image_width):
            if pixel_array[i][j] > 0 and lbl_img[i][j] == 0:
                q.enqueue((j, i))

                count = 0
                min_x, max_x, min_y, max_y = j, j, i, i

                while not q.isEmpty():
                    y, x = q.dequeue()

                    if lbl_img[x][y] == 0:
                        lbl_img[x][y] = lblid
                        count += 1


                        min_x = min(min_x, y)
                        max_x = max(max_x, y)
                        min_y = min(min_y, x)
                        max_y = max(max_y, x)

                        minMaxVals[lblid] = [min_x, min_y, max_x, max_y]

                        for p1, p2 in sides:
                            k = y + p1
                            l = x + p2
                            if 0 <= k < image_width and 0 <= l < image_height and lbl_img[l][k] == 0 and pixel_array[l][k] > 0:
                                q.enqueue((k, l))

                width = max_x - min_x + 1
                height = max_y - min_y + 1
                aspect_ratio = max(width / height, height / width)


                if aspect_ratio <= 1.8:
                    lblcnt[lblid] = count
                    lblid += 1

    sorted_dict = dict(sorted(lblcnt.items(), key=lambda a: a[1]))
    largestCompID = max(lblcnt, key=lblcnt.get)



    return lbl_img, lblcnt, largestCompID, minMaxVals

# This is our code skeleton that performs the barcode detection.
# Feel free to try it on your own images of barcodes, but keep in mind that with our algorithm developed in this assignment,
# we won't detect arbitrary or difficult to detect barcodes!
def main():

    command_line_arguments = sys.argv[1:]

    SHOW_DEBUG_FIGURES = True

    # this is the default input image filename
    filename = "Barcode2"
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
    px_array = RGBToGreyscale(image_width, image_height, px_array_r, px_array_g, px_array_b)
    px_array = ScaleAndQuantize(px_array, image_width, image_height)
    px_array = StdDev5x5(px_array, image_width, image_height)
    px_array = Gaussian3x3(px_array, image_width, image_height)
    px_array = Gaussian3x3(px_array, image_width, image_height)
    px_array = Gaussian3x3(px_array, image_width, image_height)
    px_array = Gaussian3x3(px_array, image_width, image_height)
    px_array = ThreshholdImg(px_array, 25, image_width, image_height)
    for i in range(3):
        px_array = ErosionSteps5x5(px_array, image_width, image_height)
    for i in range(4):
        px_array = DilationSteps5x5(px_array, image_width, image_height)
    lbl_img, lblcnt, largestCompID, minMaxVals = computeConnectedComponentLabeling(px_array, image_width, image_height)
    minX = minMaxVals[largestCompID][0]
    minY = minMaxVals[largestCompID][1]
    maxX = minMaxVals[largestCompID][2]
    maxY = minMaxVals[largestCompID][3]


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
    center_x = image_width / 2.0
    center_y = image_height / 2.0

    bbox_min_x = minX
    bbox_max_x = maxX
    bbox_min_y = minY
    bbox_max_y = maxY

    # The following code is used to plot the bounding box and generate an output for marking
    # Draw a bounding box as a rectangle into the input image
    axs1[1, 1].set_title('Final image of detection')
    axs1[1, 1].imshow(px_array, cmap='gray')
    rect = Rectangle((bbox_min_x, bbox_min_y), bbox_max_x - bbox_min_x, bbox_max_y - bbox_min_y, linewidth=1,
                     edgecolor='purple', facecolor='none')
    axs1[1, 1].add_patch(rect)

    # write the output image into output_filename, using the matplotlib savefig method
    extent = axs1[1, 1].get_window_extent().transformed(fig1.dpi_scale_trans.inverted())
    pyplot.savefig(output_filename, bbox_inches=extent, dpi=600)

    if SHOW_DEBUG_FIGURES:
        # plot the current figure
        pyplot.show()


if __name__ == "__main__":
    main()