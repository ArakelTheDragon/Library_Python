                                                        # Library includes
import os
import cv2                                              # image and other special formats processing library
                                                        # for computer vision
import numpy as np
from numpy import asarray
import PIL
from PIL import Image
#import VideoProcessingCore
#from VideoProcessingCore import downsample_and_save_npz

                                                        # Global variables
                                                        # Recommended: move to a separate file

path_to_your_video = "video1.mp4"
target_directory = "/home/yordan/Downloads/ToBeAnotated.4" # Recommended approach instead of using the "target"
                                                        # name directly
source_directory = "target"                             # Do not put a slash at the beginning
output_directory = "/home/yordan/Downloads/ToBeAnotated.5"                            # Do not put a slash at the beginning
output_directory3 = "target3"                             # Do not put a slash at the beginning
output_directory4 = "target4"                           # Do not put a slash at the beginning
#ErrorCode_OK  = 0                                      # Error code for everything is ok

                                                        # Function definitions
def downsample_and_save_npz(source_directory, output_directory): # Description:
                                                        # Downsample all images into resolution 128*72 and store
                                                        # each image as a .npz file

                                                        # Local variables and initialization
    # path = r'C:\Users\me\Desktop\folder'              # Source Folder
    # dstpath = r'C:\Users\me\Desktop\desfolder'        # Destination Folder
    path = source_directory  # source_directory containing the images before the processing
    dstpath = output_directory  # output_directory containing the images after the processing
                                                        # Requires libraries:
                                                        # import numpy as np
                                                        # import os
                                                        # import cv2
                                                        # import pytest
                                                        # import numpy as np

                                                        # Processing
                                                        # Test if target_directory already exists
    try:                                                # Try
        makedirs(dstpath)                               # to create target_directory
    except:                                             # if there is an error
        print("Directory already exist, images will be written in same folder") # print the error message
    files = os.listdir(path)                            # Read the files from source_directory and record them in a list
    for image in files:                                 # For index in list
        img = cv2.imread(os.path.join(path, image))     # Read the image from path + image name as an array into img
        #gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    # Convert the image into BGR2GRAY and record it into gray as
                                                        # an array

                                                        # Resize function
        src = img                                       # The name of the image to resize(source)
        dsize = (128, 72)                               # Dimension of the image after the resizing
        interpolation = cv2.INTER_LINEAR                # The method of resizing
        gray = cv2.resize(src, dsize, interpolation)

        cv2.imwrite(os.path.join(dstpath, image), gray) # Write the image gray to a file with name
                                                        # target_directory + image name

        gray = asarray(gray)                            # Convert the image to a numpy array
        output_file = os.path.join(dstpath, image)
        np.savez(output_file, gray)                     # Save the image gray to file

                                                        # Reporting
        # assert ExprectedResult == ReceivedResult


def add_progress_bar(source_directory, output_directory): # Description:
                                                        # Draw a progress bar into each image

                                                        # Local variables and initialization
    # path = r'C:\Users\me\Desktop\folder'              # Source Folder
    # dstpath = r'C:\Users\me\Desktop\desfolder'        # Destination Folder
    path = source_directory                             # source_directory containing the images before the processing
    dstpath = output_directory  # output_directory containing the images after the processing
                                                        # Requires libraries:
                                                        # import numpy as np
                                                        # import os
                                                        # import cv2
                                                        # import pytest
                                                        # import numpy as np

                                                        # Processing
                                                        # Test if target_directory already exists
    try:  # Try
        makedirs(dstpath)                               # to create target_directory
    except:                                             # if there is an error
        print("Directory already exist, images will be written in same folder") # print the error message
    files = os.listdir(path)                            # Read the files from source_directory and record them in a list
    for image in files:  # For index in list
        img = cv2.imread(os.path.join(path,image))      # Read the image from path + image name as an array into img
                                                        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    # Convert the image into BGR2GRAY and record it into gray as
                                                        # an array

                                                        # Resize function
        #src = img                                       # The name of the image to resize(source)
        #dsize = (128, 72)                               # Dimension of the image after the resizing
        #interpolation = cv2.INTER_LINEAR                # The method of resizing
        #gray = cv2.resize(src, dsize, interpolation)
                                                        # Draw a line on an image function
        start_point = 140, 700                          # The line will be drawn from pixel coordinates 0, 0
        end_point = 640, 700                            # The line will be drawn to pixel end_point,
                                                        # end_point - start_point is the width
        color = 0, 255, 0                               # The color of the line in BGR format(mode)
        thickness = 10                                  # The height or tickeness of the line
        cv2.line(img, start_point, end_point, color, thickness)
        start_point = 640, 700                          # The line will be drawn from pixel coordinates 0, 0
        end_point = 1140, 700                           # The line will be drawn to pixel end_point,
                                                        # end_point - start_point is the width
        color = 128, 128, 128                           # The color of the line in BGR format(mode)
        thickness = 10                                  # The height or thickeness of the line
        cv2.line(img, start_point, end_point, color, thickness)
                                                        # Draw a rectangle on an image function
        start_point = 500, 500                          # The rectangle will be drawn from pixel coordinates
        end_point = 600, 600                            # The rectangle will be drawn to pixel coordinates
        color = 64, 64, 64                              # The color of the rectangle in BGR format(mode)
        thickness = 10                                  # The height or thickness of the rectangle
        cv2.rectangle(img, start_point, end_point, color, thickness)
                                                        # Draw a circle on image function
        center = 640, 700                               # The circle will be drawn from pixel coordinates
        radius = 20                                     # The cicle will be drawn to radius
        color = 64, 64, 64                              # The color of the cicle in BGR format(mode)
        thickness = 2                                   # The thickness of the circle
        line = cv2.LINE_AA                              # Creates a smoother line for the circle
        cv2.circle(img, center, radius, color, thickness, line)
        cv2.imwrite(os.path.join(dstpath, image), img)  # Write the image gray to a file with name
                                                        # target_directory + image name

        #gray = asarray(gray)  # Convert the image to a numpy array
        #output_file = os.path.join(dstpath, image)
        #np.savez(output_file, gray)  # Save the image gray to file

                                                        # Reporting
        #assert ExpectedResult = ReceivedResult
def resize_and_grayscale(source_directory, output_directory): # Description:
                                                        # Convert the generated frames(images) from
                                                        # extract_video(video_path, target_dir_path) to grayscale
                                                        # and reduce their size to half

                                                        # Local variables and initialization
                                                        # Requires the libraries:
                                                        # import cv2
                                                        # import os
                                                        # import pytest

                                                        # Processing
                                                        # Reading an image in default mode
    #path = r'C:\Users\me\Desktop\folder'               # Source Folder
    #dstpath = r'C:\Users\me\Desktop\desfolder'         # Destination Folder

    path = source_directory                             # source_directory containing the images before the processing
    dstpath = output_directory                          # output_directory containing the images after the processing

                                                        # Test if target_directory exists
    """try:                                                # Try
        makedirs(dstpath)                               # to create target_directory
    except:                                             # if there is an error
        print("Directory already exist, images will be written in same folder") # print the error message"""

    files = os.listdir(path)                            # Read the files from source_directory and record them in a list

    for image in files:                                 # For index in list
        img = cv2.imread(os.path.join(path, image))     # Read the image from path + image name as an array into img
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # Convert the image into BGR2GRAY and record it into gray as
                                                        # an array
                                                        # Resize function cv2.resize
        #src = gray                                      # The name of the image to resize(source)
        #dsize = (640, 360)                              # Dimenstion of the image after the resizing
        #interpolation = cv2.INTER_LINEAR                # The method of resizing
        #gray = cv2.resize(src, dsize, interpolation)
        cv2.imwrite(os.path.join(dstpath, image), gray) # Write the image gray to a file with name
                                                        # target_directory + image name

                                                        # Displaying the image
    #cv2.imshow(window_name, image)


def extract_video(video_path, target_dir_path):         # Description:
                                                        # Extract video frames as JPG images named 00001.jpg,
                                                        # 00002.jpg... to a target directory, use pip install
                                                        # opencv-python package

                                                        # Local variables and initialization
                                                        # Requires the libraries:
                                                        # import cv2
                                                        # import os
    #global ErrorCode_OK                                 # In order to use global variables, they must be defined as
                                                        # global, unnecessary to use error codes if we use assert

                                                        # Processing
    vidcap = cv2.VideoCapture(video_path)               # load the video from video_path
    success, image = vidcap.read()                      # success = 1 if we can read the first frame
    count = 0                                           # coun = 0 for the beginning of the cycle
    while success:                                      # While success == 1
        cv2.imwrite(os.path.join(target_dir_path, f"{count:05}.jpg"), image) # save frame as JPEG file
        success, image = vidcap.read()                  # success == 1 if there is another frame to read
        count += 1                                      # count = count + 1

    assert success == 0 and count == 300                # If there are no more frames to read, success = 0
                                                        # if we have read 299 frames count = 300

                                                        # Reporting
    #return ErrorCode_OK                                 # Error code for everything is ok

if __name__ == "__main__":                              # Description:
                                                        # Will run the test functions

                                                        # Local variables and initialization

                                                        # Processing
    #downsample_and_save_npz(target_directory, output_directory4)
    #add_progress_bar("target", "target3")
    resize_and_grayscale(target_directory, output_directory)
    #extract_video(path_to_your_video, "target")         # Path to video, target directory

                                                        # Processing

                                                        # Reporting
