# Library includes
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.pyplot import imshow, figure
import numpy as np  # Needed for remove_blur
import scipy.ndimage
from scipy import ndimage, misc
from scipy.ndimage.filters import gaussian_filter  # Needed for xdog, dog and hatch
import cv2  # Needed for xdog, dog, hatch and remove_blur
import pytest  # Needed for the PyTest testing framework
import requests  # Needed for get_links_in_a_url
import re  # Needed for get_links_in_a_url
import vlc  # Needed by media_player_vlc
import time  # Needed by media_player_vlc
import math  # Needed for find_string_in_index
import os  # Needed for find_string_in_index
import glob  # Needed for reading_in_an_image
import sys
import shutil


def Weighten_image(Gaussian_blurred_image):
    Result = cv2.addWeighted(Gaussian_blurred_image, 1.5, Gaussian_blurred_image, -0.5, 0, Gaussian_blurred_image)
    return Result

def Image_filter2d(Image):
    # Apply blurring kernel
    kernel2 = np.ones((5, 5), np.float32) / 25
    Result = cv2.filter2D(src=Image, ddepth=-1, kernel=kernel2)
    return Result


def Image_edge_fix(Image):
    dst = cv2.edgePreservingFilter(Image, flags=1, sigma_s=60, sigma_r=0.4)
    return dst


def compose(*image_list):
    # Error checking
    if len(image_list) == 0:
        raise Exception("The list should not be empty!")

    background = image_list[0].copy()

    for image in image_list[1:]:
        # Take only region of logo from logo image.
        indices = np.all(image != [0, 0, 0], axis=-1)
        print(indices)
        background[indices] = image[indices]
    return background


def paste_image(s_img, l_img, x_offset, y_offset):
    l_img[y_offset:y_offset + s_img.shape[0], x_offset:x_offset + s_img.shape[1]] = s_img
    # print("Breakpoint")


def equalize_histogram_rgb(rgb_img):
    # convert from RGB color-space to YCrCb
    ycrcb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2YCrCb)

    # equalize the histogram of the Y channel
    ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])

    # convert back to RGB color-space from YCrCb
    equalized_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)

    return equalized_img


def colorize_mask(image, color=[255, 0, 0], color_to_be_replaced=[255, 255, 255]):  # [255, 0, 0] = red in RGB format
    img1 = image.copy()
    white_pixels = np.where(img1 == color_to_be_replaced)
    img1[white_pixels[0], white_pixels[1], :] = color
    return img1


def clear_directory(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def save_image_to_file(image, path):
    print("save_image_to_file = ", image.shape, path)
    cv2.imwrite(path, image)  # Write the image gray to a file with name


def make_empty_rgb_image(width, height, color=(0, 0, 0)):
    print("make_empty_rgb_image = ", width, height, color)
    blank_image = np.zeros((height, width, 3), np.uint8)
    return blank_image


def gray2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


def binary_not(image):
    # Invert the colors of the image

    inverted_dst = image.astype(np.uint8)
    inverted_image = cv2.bitwise_not(inverted_dst)
    return inverted_image


def binary_or(*image_list):
    if len(image_list) == 0:
        raise Exception("The list should not be empty!")

    background = image_list[0].copy()
    background_dst = background.astype(np.uint8)

    for image in image_list[1:]:
        # Take only region of logo from logo image.
        image_dst = image.astype(np.uint8)
        background_dst = cv2.bitwise_or(background_dst, image_dst)

    return background_dst


def binary_and(*image_list):
    if len(image_list) == 0:
        raise Exception("The list should not be empty!")

    background = image_list[0].copy()
    background_dst = background.astype(np.uint8)

    for image in image_list[1:]:
        # Take only region of logo from logo image.
        image_dst = image.astype(np.uint8)
        background_dst = cv2.bitwise_and(background_dst, image_dst)

    return background_dst


"""def binary_and(img2, img1):
    # Take only region of logo from logo image.
    dst2 = img2.astype(np.uint8)
    dst1 = img1.astype(np.uint8)
    dst2_fg = cv2.bitwise_and(dst2, dst1)
    return dst2_fg"""


def binary_xor(*image_list):
    if len(image_list) == 0:
        raise Exception("The list should not be empty!")

    background = image_list[0].copy()
    background_dst = background.astype(np.uint8)

    for image in image_list[1:]:
        # Take only region of logo from logo image.
        image_dst = image.astype(np.uint8)
        background_dst = cv2.bitwise_xor(background_dst, image_dst)

    return background_dst


def outline(image, thickness=1, colour="#FFFFFF"):
    """filter: outline - it will get a mask image and a keyword argument (outline thickness) and a color argument.
    It will dilate the source mask image by the thickness (e.g. thickness = 2) and it will XOR or DIFF the dilated
    image and the original image, leaving only a thin outline around the original mask. The oputline is then drawn
    by the specific color."""
    dilated_image = dilate(image, kernel_size=(thickness, thickness))
    outline_image = binary_xor(dilated_image, image)  # dilated_image and original image
    # coloured_image = colour_the_image() # Will be implemented later and return the coloured image
    return outline_image


def convert_hash_to_bgr_color(string="#000000"):
    rgb_channel_red = "0x" + string[1:3]
    rgb_channel_green = "0x" + string[3:5]
    rgb_channel_blue = "0x" + string[5:7]

    return [int(rgb_channel_blue, base=16), int(rgb_channel_green, base=16), int(rgb_channel_red, base=16)]


def convert_hash_to_rgb_color(string="#000000"):
    rgb_channel_red = "0x" + string[1:3]
    rgb_channel_green = "0x" + string[3:5]
    rgb_channel_blue = "0x" + string[5:7]

    return [int(rgb_channel_red, base=16), int(rgb_channel_green, base=16), int(rgb_channel_blue, base=16)]


def color_to_mask(image, color=[0, 255, 0], new_color=[0, 0, 0]):
    # [255, 0, 0] = red in RGB format, in some IDEs the
    # channels are reversed, BGR not RGB
    # Replaces all colors except the designated one
    # Replace all non needed colors in color_to_be_replaced by color
    # Replaces all colors except the designated one
    # Replace all non needed colors in color_to_be_replaced by color
    img1 = image.copy()
    other_pixels = np.where(img1 != color)
    elected_pixels = np.where(img1 == color)
    img1[elected_pixels[0], elected_pixels[1], :] = [255, 255, 255]
    img1[other_pixels[0], other_pixels[1], :] = new_color
    return img1


def split_string_to_tuple(string_value):
    resulting_tuple = tuple(map(str, string_value.split(', ')))  # The elements of the tuple must always be strings
    # for example ["string1", "string2"] or change
    # str to int
    return resulting_tuple


def remove_blur(image):
    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpen = cv2.filter2D(image, -1, sharpen_kernel)
    return sharpen


def image_improve_quality(image):
    # the default
    image_file.save("improved_quality.jpg", quality=95)  # 95% is the maximum quality


def save_to_file(new_filename, old_filename):
    cv2.imwrite(new_filename, old_filename)


def create_random_tuple():  # Description:

    # Create a list of strings to use with the string functions

    # Local variables and initialization:
    # Requires the libraries:
    # import os, math
    # import pytest
    # import numpy as np
    # from string import asciii_lowercase, digits
    # import random
    # from random import choice

    # Processing
    chars = ascii_lowercase + digits
    lst = [''.join(choice(chars) for _ in range(2)) for _ in range(100)]  # list = join random choice characters
    # for a cycle of 2 characters and a list of 100 elements by 2
    # characters

    # Reporting
    return lst  # Return the list


def HowTo_OutputAnArrayInPython():
    x = np.linspace(0, 10, 100)  # Return evenly spaced numbers over an interval,
    # the interval is 0 to 10, the points are 100
    x  # Show the resulting array

    plt.rcParams['figure.figsize'] = [10, 5]  # Create a figure of size 10, 5
    plt.plot(x, np.sin(x))  # Plot on the x axis, the sin of the resulting array
    plt.plot(x, np.cos(x))  # Plot on the x axis, the cos of the resulting array
    # ??? The figure will have on the x and y, the values of the
    # array, not the figure size ???

    return plt.show()  # plt.show shows the image from the resulting array, but
    # we must always have a return statement

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(20, 7))  # plt.subplots (rows, columns, sharex, sharey True
    # or 'all' for all x or y asix will be shared among all plots,
    # squeeze, subplot_kwdict, gridspec_kwdict, fig_kw,
    # figure size)
    # documentation for the function is at:
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html
    # The variables figFigure and axes record the figure and axes

    axes[0].plot(x, x)  # Plot the the resuting array on the x axis
    axes[0].set_title('X Raw Data', size=22)  # Set the title of the plot
    axes[1].plot(x, np.cos(x))  # Plot the cos of the resulting array on the x axis
    axes[1].set_title('CosX Data', size=22)  # Set the title of the plot
    axes[2].plot(x, np.sin(x))  # Plot the sin of the resulting array on the x asix
    axes[2].set_title('SinX Data', size=22)  # Set the title of the plot

    # display image by IPython
    pil_img = Image(filename='target/02_mandelbrot.png')
    display(pil_img)

    # display image by Pillow
    # % matplotlib inline
    pil_im = Image.open('target/02_mandelbrot.png', 'r')
    imshow(np.asarray(pil_im))

    # display image by opencv
    img = cv2.imread('target/02_mandelbrot.png')
    img2 = img[:, :, ::-1]
    plt.imshow(img)

    rng = np.random.RandomState(0)
    for marker in ['o', '.', ',', 'x', '+', 'v', '^', '<', '>', 's', 'd']:
        plt.plot(rng.rand(5), rng.rand(5), marker,
                 label="marker='{0}'".format(marker))
        plt.legend(numpoints=1)
        plt.xlim(0, 1.8);


def cmplx(real, im):  # There is no function in cmath to do this
    # Local variables and initialization
    # Requires the libraries:
    # none

    # Processing
    print("DEBUG: cmplx = ")
    print(complex(real, im))
    print("DEBUG: cmplx = ")

    # Reporting
    return real, im * (-1)  # The j, has to be added separately


def cmplx_add(c1, c2):
    # Local variables and initialization
    # Requires the libraries:
    # import cmath
    # import math
    print(c1.real + c2.real, c1.imag + c2.imag)
    return  # (c1[0]+c1[1],c1[1]+c2[1])               # Does not work as the object is not subscribtable


def cmplx_abs(c):
    # Local variables and initialization
    # Requires the libraries:
    # import cmath
    # import math
    print(abs(c))
    return  # math.sqrt(c[0]*c[0] + c[1]*c[1])        # Does not work as the object is not subscribtable


def cmplx_mult(c1, c2):
    # Local variables and initialization
    # Requires the libraries:
    # import cmath
    # import math
    # a, b = c1                                      # Does not work, can not unpack non-iterable complex object
    # c, d = c2                                      # Does not work, can not unpack non-iterable complex object
    a = c1.real
    b = c1.imag
    c = c2.real
    d = c2.imag
    print("DEBUG: cmplx_mult calls cmplx = ")
    print(cmplx(a * c - b * d, a * d + b * c * (-1)))
    print("DEBUG: cmplx_mult = ")
    return complex(a * c - b * d, a * d + b * c)


def cmplx_square(c):
    return cmplx_mult(c, c)


def ReturnTheImaginaryAndRealPart(real, im):  # There is no function in cmath to do this
    return (real, im)  # Return the real and imaginary part of a complex number

    # Implementation of matplotlib.pyplot.annotate() function


def AnnotateByMatplotlob():
    # Local variables and initialization
    # Required libraries:
    # import numpy as np
    # import matplotlib .pyplot as plt

    # Processing
    t = np.linspace(-10, 10, 100)  # Return evenly spaced numbers over a specified interval,
    # the interval is -10, 10, the points are 100
    sig = 1 / t  # T = 1/F, F = 1/T

    plt.axhline(y=0, color="green", linestyle="--")  # Add a horizontal line accros the axix, y is the
    # coordinate on the y axis, color is the color, linestyle is
    # the line type
    plt.axhline(y=0.5, color="green", linestyle=":")  # Add a horizontal line accros the axis, y is the
    # coordinate on the y axis, color is the color, linestyle is
    # the line type
    plt.axhline(y=1.0, color="green", linestyle="--")  # Add a horizontal line accros the axis, y is the
    # coordinate on the y axis, color is the color, linestyle is
    # the line type

    plt.axvline(color="black")  # Add a vertical line accros the axes, the color = "black"

    plt.plot(t, sig, linewidth=2,
             label=r"$\sigma(t) = \frac{1}{x}$")

    plt.xlim(-10, 10)
    plt.xlabel("t")
    plt.title("Graph of 1 / x")
    plt.legend(fontsize=14)
    # Reporting
    # plt.show()                                     # There is no need to add plt.show () separately, its
    return plt.show()  # already in the return and will display the image


def SinWaveByNumpy():
    # Local variables and initialization
    # Requires libraries:
    # import numpy as np
    # import matplotlib .pyplot as plt

    Time = np.arange(0, 10, 0.1)  # We are creating an array named time, of type numpy,
    # this will be the x axis or the time of the sin wave,
    # we arrange the array from 0 to 10, 0.1 per incrementation

    Amplitude = np.sin(Time)

    plt.plot(Time, Amplitude)  # Plot a sin wave, using the time and amplitute we
    # obtained for the sin wave
    # Give a title for the sine wave plot
    plt.title('Sine wave')

    # Give the x axis a label for the sine plot
    plt.xlabel('Time')

    # Give the y axis label for the sine plot
    plt.ylabel('Amplitude = sin(time)')

    # Give the grid
    plt.grid(True, which='both')

    # Adds a horizontal line accros the axis
    plt.axhline(y=0, color='k')

    # Show the sine wave
    plt.show()

    # Reporting
    return plt.show()


def GraphicalRepresentationOfSound():
    # Local variables and initialization
    x = []  # Define an empty array with unknown elements

    # Processing
    for i in range(0, 256, 1):  # Start from 0, go to 256, step is 1
        x.append(i)  # Append to the array x, the value of i for all elements
        # print (x)                                  # Display the value of x to the screen

    plt.rcParams['figure.figsize'] = [10, 5]  # Create the figure for the graphics

    plt.plot(x, np.sin(x))  # Plot the value of x as a function of sin x
    plt.plot(x, np.cos(x))  # Plot the value of x as a function of cos x

    plt.show()  # Show the graphics

    # Reporting
    return plt


def SineWaveRepresentation():
    # Local variables and initialization
    x = np.linspace(0, 10, 100)  # Return evenly spaced numbers over an interval, the interval
    # is 0, 10, the points are 100

    # Processing
    plt.rcParams['figure.figsize'] = [10, 5]

    plt.plot(x, np.sin(x))
    plt.plot(x, np.cos(x))

    plt.show()  # Show the image

    # Reporting
    return plt.show()


def FunctionOfX():
    # Local variables and initialization
    x = np.linspace(0, 10, 100)

    # Processing
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(20, 7))
    axes[0].plot(x, x)
    axes[0].set_title('X Raw Data', size=22)
    axes[1].plot(x, np.cos(x))
    axes[1].set_title('CosX Data', size=22)
    axes[2].plot(x, np.sin(x))
    axes[2].set_title('SinX Data', size=22)

    # Reporting
    return axes


def DisplayAndRecordAnImageToAFile():
    # Local variables and initialization
    # Requires the libraries:
    # from IPython.display import Image

    # Processing
    # display image by IPython
    pil_img = Image(filename='target/02_grey.png')  # Record the image to this file, this function does not work
    display(pil_img)  # Display the image

    # Reporting
    return pil_img  # This line will show the grey image again


def create_palindrome(string1):  # Description:
    # A palindrome is spelled the same way forward and reverse
    # Local variables and initialization:
    # Requires the libraries:
    # import os, math
    # import pytest
    # Processing
    if list(string1) != list(rev):
        print("NOT PALINDROME")
        return -1
    else:
        return 0


def find_string_in_index(string1, string2):
    # Description:
    # Find str1 in str2
    string_index = string2.find(string1)
    return string_index  # Return the first position at which string1 is found in


# string2


# Description:
# Repeat a string from var1 to var2 n times
# no commas and so on
def repeat_a_string(str1, n):
    return str1 * n  # Return the string n times


def Cicles():
    x = []  # Define an array with unknown elements
    for i in range(0, 256, 1):  # for (i = 0; i < 256; i+1)
        ArrayName.append(value)

    return ArrayName


def list_files_in_directory(path):  # path = "images/"
    list_of_files = os.listdir(path)  # If its in a function, it will not show
    # the output
    return list_of_files


def reading_in_an_image(path_to_read):  # Reads the images in path_to_read = images/*
    input_images = [cv2.imread(img_path) for img_path in sorted(glob.glob(path_to_read))]  # Returns the sorted files
    return input_images


def get_links_in_a_url(url):  # url = http://www.google.com
    html = requests.get(url).text
    links = re.findall('"(https?://.*?)"', html)
    return links


def media_player_vlc():
    # creating vlc media player object
    media_player = vlc.MediaPlayer("1.mp4")

    # start playing video
    media_player.play()

    # wait so the video can be played for 5 seconds
    # irrespective for length of video
    time.sleep(5)


def is_string_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False


def is_string_integer(string):
    try:
        int(string)
        return True
    except ValueError:
        return False


def convert_to_correct_type(string):
    if string == "true":
        return True
    if string == "false":
        return False
    if is_string_float(string):
        return float(string)
    if is_string_integer(string):
        return int(string)
    if isinstance(string, str):
        return string
    return None


def draw_lines_to_image(img, segments, min_width=None, min_length=None):
    for i in range(segments.shape[0]):
        pt1 = (int(segments[i, 0]), int(segments[i, 1]))
        pt2 = (int(segments[i, 2]), int(segments[i, 3]))
        length = math.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (pt1[1] - pt2[1]) * (pt1[1] - pt2[1]))
        width = segments[i, 4]
        if (min_length is None) or (length
                                    >= min_length):
            if (min_width is None) or (width.item() >= min_width):
                cv2.line(img, pt1, pt2, (255, 0, 0), int(np.ceil(width / 2)))
                cv2.line(img, pt1, pt2, (0, 0, 255), 2)


def alpha_blending(foreground, background, alpha=0.5):
    dst = foreground * alpha + background * (1 - alpha)
    dst = dst.astype(np.uint8)
    return dst


def hatch(image):
    """
    A naive hatching implementation that takes an image and returns the image in
    the style of a drawing created using hatching.
    image: an n x m single channel matrix.
    returns: an n x m single channel matrix representing a hatching style image.
    """
    xdogImage = xdog(image, 0.1)

    hatchTexture = cv2.imread('./textures/hatch.jpg', cv2.CV_LOAD_IMAGE_GRAYSCALE)

    height = len(xdogImage)
    width = len(xdogImage[0])

    if height > 1080 or width > 1920:
        print("This method only supports images up to 1920x1080 pixels in size")
        sys.exit(1)

    croppedTexture = hatchTexture[0:height, 0:width]

    return xdogImage + croppedTexture


def xdog(image, epsilon=0.01):  # Accepts only grayscale images
    """
    Computes the eXtended Difference of Gaussians (XDoG) for a given image. This
    is done by taking the regular Difference of Gaussians, thresholding it
    at some value, and applying the hypertangent function the the unthresholded
    values.
    image: an n x m single channel matrix.
    epsilon: the offset value when computing the hypertangent.
    returns: an n x m single channel matrix representing the XDoG.
    """
    phi = 10

    difference = dog(image, 200, 0.98) / 255
    diff = difference * image

    for i in range(0, len(difference)):
        for j in range(0, len(difference[0])):
            if difference[i][j] >= epsilon:
                difference[i][j] = 1
            else:
                ht = np.tanh(phi * (difference[i][j] - epsilon))
                difference[i][j] = 1 + ht

    return difference * 255


def dog(image, k=200, gamma=1):
    """
    Computes the Difference of Gaussians (DoG) for a given image. Returns an image
    that results from computing the DoG.
    image: an n x m array for which the DoG is computed.
    k: the multiplier the the second Gaussian sigma value.
    gamma: the multiplier for the second Gaussian result.

    return: an n x m array representing the DoG
    """

    s1 = 0.5
    s2 = s1 * k

    gauss1 = gaussian_filter(image, s1)
    gauss_temp = gaussian_filter(image, s2)
    gauss2 = gamma * gauss_temp

    differenceGauss = gauss1 - gauss2
    return differenceGauss


def threshold(image, thresh=127, maxval=255, type=cv2.THRESH_BINARY):  # Accepts only grayscale images
    ret, thresh1 = cv2.threshold(image, thresh=thresh, maxval=maxval, type=type)
    return thresh1


def sobel(grayscaled_image, axis=-1):  # Accepts only grayscale images
    return scipy.ndimage.sobel(grayscaled_image, axis=axis)


def laplace(grayscaled_image, ddepth=cv2.CV_16S, kernel_size=3):  # Accepts only grayscale images
    dst = cv2.Laplacian(grayscaled_image, ddepth, ksize=kernel_size)
    return dst


def gaussian_blur(img, kernel_size=(5, 5)):  # Accepts only grayscale images
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, kernel_size, 0)


def edge_canny(img, low_threshold=100, high_threshold=200):  # Accepts only grayscale images
    edges = cv2.Canny(img, low_threshold, high_threshold)
    return edges


def erode(grayscaled_image, kernel_size=(5, 5)):  # Accepts only grayscale images
    kernel = np.ones(kernel_size, 'uint8')
    eroded_image = cv2.erode(grayscaled_image, kernel, iterations=1)
    return eroded_image


def dilate(img, kernel_size=(5, 5)):  # Accepts only grayscale images
    kernel = np.ones(kernel_size, 'uint8')
    dilate_img = cv2.dilate(img, kernel, iterations=1)
    return dilate_img


def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def showimage(img, cmap=None, figsize=[12, 12]):
    figure(figsize=figsize)
    imshow(np.asarray(img), cmap=cmap)


def resize(image, width, height):  # Description:
    interpolation = cv2.INTER_LINEAR  # The method of resizing
    output_image = cv2.resize(image, (width, height), interpolation)
    return output_image
