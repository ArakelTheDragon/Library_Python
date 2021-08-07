# Please consider donating
# https://www.paypal.com/donate?hosted_button_id=JZXRFTC9BPWTN&source=url








# Library includes
import matplotlib.pyplot as plt
# from mpl_toolkits import mplot3d
# from PIL import Image
# import numpy as np
# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
# from matplotlib.figure import Figure
import cv2
from utils import grayscale


# Function definitions
"""def compare_tuples(tuple1, tuple2):
    if tuple1[0] == tuple2[0] and tuple1[1] == tuple2[1] and tuple1[2] == tuple2[2]:
        return True

    return False"""

def compare_tuples(tuple1, tuple2):
    if tuple1 < tuple2:
        return False

    return True


# Function calls
im = cv2.imread("/media/arakel-ubuntu-ssd/Data/Python assignments/Tutorials/1.jpg")  # Read the image in directory "samples/00003_09.jpg" into a numpy nd array
im = grayscale(im)
cv2.imwrite("image_grayscale.png", im)  # we record the new image with the replaced colors in a name "image.png" from array im,
print("Debug im.shape =", im.shape)
# result = im.copy()
# px = im.copy()
"""expected_colors = [(120,121,119), (120,121,119), (120,121,119), (120,121,119), (120,121,119), (120,121,119), (121,122,120), (121,122,120), (122,123,121), (122,123,121), (122,123,121), (122,123,121), (123,121,120), (123,121,120), (125,121,120), (125,121,120), (125,121,120), (125,121,120), (125,121,120), (125,121,120), (126,122,121), (126,122,121), (126,122,121), (126,122,121), (126,122,121), (126,122,121), (126,122,121), (126,122,121), (123,122,118), (123,122,118), (123,122,118), (123,122,118), (122,121,117), (122,121,117), (122,121,117), (122,121,117), (122,121,117), (121,120,116), (121,120,116), (121,120,116), (121,120,116), (120,119,115), (120,119,115), (119,118,114), (119,118,114), (118,117,113), (118,117,113), (118,117,113), (118,117,113), (118,117,113), (118,117,113), (119,118,114), (120,119,115), (121,120,116), (121,120,116), (122,121,117), (121,120,116), (122,121,117), (122,121,117), (122,121,117), (122,121,117), (122,121,117), (121,120,116), (121,120,116), (120,119,115), (120,119,115), (119,118,114), (119,118,114), (120,118,117), (120,118,117), (120,118,117), (119,117,116), (120,118,117), (120,118,117), (122,118,117), (122,118,117), (122,117,118), (122,117,118), (122,117,118), (122,117,118), (122,118,117), (122,118,117), (123,118,117), (123,118,117), (124,119,118), (125,120,119), (124,120,119), (124,120,119), (124,120,119)]"""  # Colors to be replaced in the image

expected_colors = [(168)]
replacing_color = (0)  # Sometimes the R, G, B channels are switched to B, G, R on some platforms,
# color to replace the expected colors

# The lists are needed for the ax scatter function
x = []
y = []
z = []
c = []

width, height = im.shape  # Get the width and height of the image, without the channels = _, for a grayscale image
                          # there are no channels
for row in range(0, height):  # For row in range 0 to maximum,
    for col in range(0, width):  # for col in range 0 to maximum,
        pix = im[col, row]  # get the pixel on coordinate im[col, row],
        print(pix)  # print the pixel in RGB or BGR format (127, 127, 127)
        NewCol = (pix / 255)  # Scale the 3 channels to a 0-1 interval

        if (not NewCol in c):  # if the color is not already in the c list,
            x.append(pix)  # append the first channel,
            #y.append(pix[1])  # append the second channel,
            #z.append(pix[2])  # append the third channel,
            c.append(NewCol)  # append c with the list of colors,

        # In this block other pixel colors can be removed too for every pixel
        for color in expected_colors:  # for index in the expected colors list,
            if compare_tuples(pix, color)==False:  # if the color is in the expected colors list,
                pix = replacing_color  # pix equals the replacing color,
            im[col, row] = pix  # we change the expected color with the replacing color in the image,

cv2.imwrite("image.png", im)  # we record the new image with the replaced colors in a name "image.png" from array im,

