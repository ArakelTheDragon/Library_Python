# Library includes
import os
import argparse
from topgisviz.mosaic_generator import process

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        help="Path to the config file."
    )

    args = parser.parse_args()
    config = args.config

    process(config) # Calls the function(fle) process.py from mosaic_generator
                    # Pass the config file to the function
"""
TopGIS Dataset Visualization

In this assignment, you will work with satellite images downloaded along a GPS route from TopGIS service. You can download the dataset here: http://download.mykkro.cz/valeo/route.zip .

1. Create a new Python module (package) named topgis-viz. This module will have this basic directory structure:

topgis-viz/
    config/     # this will contain your YAML configuration files (if any is necessary)
        config.yaml    # a default configuration file
    data/       # unpack the dataset into this directory (so it contains a subdirectory named topgis with images)
    target/     # this folder will contain all of your outputs
    topgisviz/  # this folder will contain the module code
        __init__.py
        # all the other stuff
    setup.py    # the setup script for the package
    requirements.txt
    README.md   # a documentation page, with a short description of the module and usage instructions
    topgis-test.py
    .gitignore  # don't forget to .gitignore data/ and target/ folders

You will use the module by calling it from command line as follows:

python topgis-test.py [--config=config/config.yaml]

The --config parameter is optional, if not used, the default value config/config.yaml will be used. Use argparse library to read the command line arguments, like this:

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        help="Path to the config file."
    )

    args = parser.parse_args()
    config = args.config

When run, this script will read the configuration file in YAML format (read about YAML here: https://yaml.org/) into a Python dictionary. You can use this code:

def load_yaml(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as infile:
        return yaml.load(infile, Loader=yaml.FullLoader)

The configuration file will look similarly to this:

# configuration file for topgis-viz

# glob pattern for finding the source files
sources: "data/topgis/utm_*.jpg"
output_dir: "target/mosaic/"
mosaic:
    columns: 5
    rows: 2
    cell_width: 250
    cell_height: 250
pipeline:
    - "A0 = src"
    - "A1 = greyscale A0"
    - "A2 = red A0"
    - "A3 = green A0"
    - "A4 = blue A0"
    - "B0 = equalized A0"
    - "B1 = equalized_greyscale A1"
    - "B2 = equalized A2"
    - "B3 = equalized A3"
    - "B4 = equalized A4"

These parameters govern what the script will do with the images. The script will:

    get a list of available images (based on sources parameter, via glob.glob function) and sort it alphabetically in increasing order
    iterate over the list of images and for each of those source images:
        create an empty RGB image with specified number of cells (rows, columns)
        the cells of the grid are denoted by letter-number code indicating column and row. E.g. A0 is top-left cell.
        run a processing pipeline on the source image. The pipeline is defined by a list of strings, which are evaluated in sequence. The format of each string is as follows: CELL = FUNCTION [ARGS]. E.g. string B0 = equalized A0 means that cell B0 of the grid will be filled with an image that comes as result of calling image function equalized on an image in cell A0.
        the original size of the images is 1000x1000 pixels. Do the pipeline operations on images with this original size. Resize the images (to cell_width * cell_height) only when putting them into the big mosaic image.
        the result of the pipeline will be stored in the specified output directory as a JPG image (Use this naming pattern: f"{index:05}.jpg").

The following pipeline functions should be supported:

    src - just the source image
    greyscale RGB_IMAGE - returns a greyscale image constructed from RGB_IMAGE
    red RGB_IMAGE - returns a color image that contains only red channel
    green RGB_IMAGE - returns a color image that contains only green channel
    blue RGB_IMAGE - returns a color image that contains only blue channel
    equalized RGB_IMAGE - returns a color image with equalized intensities
    equalized_greyscale BW_IMAGE - returns a greyscale image with equalized intensities

Most of the functions should be easy to implement. For equalization of colored images, use this function:

def equalize_histogram_rgb(rgb_img):
    # convert from RGB color-space to YCrCb
    ycrcb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2YCrCb)

    # equalize the histogram of the Y channel
    ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])

    # convert back to RGB color-space from YCrCb
    equalized_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)

    return equalized_img

The cv2.equalizeHist function can be used for equalization of black and white images.

Running this script should result with the target/mosaic directory populated with 624 of JPEG images named 00000.jpg to 00623.jpg. Each of these images should be 1250 pixels wide and 500 pixels high and contain a mosaic of images, first row containing original, greyscale, red, green and blue images and the second row should contain the equalized versions of these images.
"""