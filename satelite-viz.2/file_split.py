                                                        # Library includes
import os
import cv2                                              # image and other special formats processing library
                                                        # for computer vision
import glob
import numpy as np
import random

                                                        # Global variables
                                                        # Recommended: move to a separate file

                                                        # Recommended approach instead of using the "target"
                                                        # name directly
source_pattern = "data/utm_*.jpg"                       # Do not put a slash at the beginning
output_directory = "target"                             # Do not put a slash at the beginning

                                                        # Function definitions
def cutting(pattern, dstpath):                          # Description:

    files = glob.glob(pattern)                          # Read the files from source_directory and record them in a list
    print(files)

    for relative_path in files:                         # For index in list
        img = cv2.imread(relative_path) # Read the image from path + image name as an array into img
        filename = os.path.basename(relative_path)
        file_id = filename.replace("utm_", "").replace(".jpg", "")
        print(filename)

        for crop in range(12):                          # For index in range
            cropped_image_name = f"{file_id}_{crop:02}.jpg"
            cropped_image_path = os.path.join(dstpath, cropped_image_name)

            w = img.shape[0]
            h = img.shape[1]
            cx = int(w / 2)
            cy = int(h / 2)
            dx = int(random.random() * 300 - 150)
            dy = int(random.random() * 300 - 150)
            sx = random.random() * 100 + 100
            sy = random.random() * 100 + 100
            crop_image = img[cx + dx - int(sx / 2):cx + dx + int(sx / 2), cy + dy - int(sy / 2):cy + dy + int(sy / 2)]
                                                        # New snippet END

            cv2.imwrite(cropped_image_path, crop_image) # Write the image crop_img to a file with name
                                                        # target_directory + image name

                                                        # Displaying the image
        #cv2.imshow("Test", crop_img)                   # Show the image


cutting(source_pattern, output_directory)