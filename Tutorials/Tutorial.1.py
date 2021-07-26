# Library imports
from utils import showimage, gaussian_blur, reading_in_an_image, save_image_to_file, laplace
import cv2
from cv2 import GaussianBlur
import numpy as np
import os # Needed for os.path.join

# Global variables
Image_directory = 'Data/*'

# Function definitions


Input_images = reading_in_an_image(Image_directory) # Read the image list into an numpy ndarray
Result1 = gaussian_blur(Input_images[0])
Result2 = Weighten_image(Result1)
#Result3 = Image_filter2d(Result2)
#Result4 = alpha_blending(Result3, Result3)
#Result5 = laplace(Result2)
Result6 = Image_edge_fix(Input_images[0])
Result8 = cv2.detailEnhance(Result2
                            , sigma_s=200, sigma_r=1) # sigma_s is kernel, signma_r is range
# The variable 'kernel' determines how big the neighbourhood of pixels must be to perform filtering. Its range is from: 0 - 200.
# The variable 'range' determines how the different colours within the neighbourhood of pixels will be averaged with each other. Its range is from: 0 - 1.
#dst_gray, dst_color = cv2.pencilSketch(Result6, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
Result9 = cv2.stylization(Input_images[0], sigma_s=60, sigma_r=0.07)
save_image_to_file(Result9, 'Stylization'
                            '.jpg')
#save_image_to_file(dst_gray, 'dst_gray.jpg') # Save the image to a file, numpy ndarray, image name
#save_image_to_file(dst_gray, 'dst_color.jpg') # Save the image to a file, numpy ndarray, image name