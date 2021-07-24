# Description
"""These parameters govern what the script will do with the images. The script will:

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
"""
# Library includes
import os
import numpy as np
import cv2
import yaml
import math
import glob
from pylsd import lsd
from ciirc_utils import resize, grayscale, showimage, gaussian_blur, threshold, xdog, sobel, laplace, draw_lines_to_image, erode, dilate, convert_to_correct_type, alpha_blending, save_image_to_file, make_empty_rgb_image
from ciirc_utils import clear_directory, gray2rgb, binary_and, binary_xor, outline, convert_hash_to_bgr_color, color_to_mask, colorize_mask, equalize_histogram_rgb, paste_image

# Global variables

def process(path_to_configuration_file):
    loaded_yaml = load_yaml(path_to_configuration_file)
    clear_directory(loaded_yaml["output_dir"])
    process_config(loaded_yaml)


def load_yaml(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as infile:
        return yaml.load(infile, Loader=yaml.FullLoader)


def process_config(config):
    print(config)
    file_list = get_list_of_available_images(config["sources"])
    for file_name in file_list[:1]: # Only first 1 elements will be processed, from 0 to 1, from 1 to 0 is 1:
        generate_mosaic(file_name, config)


def get_list_of_available_images(source_pattern):
    return sorted(glob.glob(source_pattern))


def generate_mosaic(file_path, config):
    print("generate_mosaic = ", file_path, config)
    width = config["mosaic"]["columns"] * config["mosaic"]["cell_width"] # Multiply mosaic by columns
    height = config["mosaic"]["rows"] * config["mosaic"]["cell_height"] # Multiply mosaic by rows
    mosaic = make_empty_rgb_image(width, height)
    print("generate_mosaic = ", mosaic.shape)
    source_image = cv2.imread(file_path)
    context = { "cell_width": config["mosaic"]["cell_width"], "cell_height": config["mosaic"]["cell_height"]} # Dictionary
    process_pipeline(config["pipeline"], source_image, mosaic, context) # mosaic is the output image
    file_name = os.path.basename(file_path) # Take the file name from the directory
    file_id = file_name.replace(".jpg", "").replace("utm_", "") # Replace string1 with string2, second time replace string1 with string2
    filename_new = f"{file_id:5}.jpg"  # Using 05 gives an error
    target_directory = config["output_dir"] # Taken from the .yaml file
    target_file_path = os.path.join(target_directory, filename_new)
    save_image_to_file(mosaic, target_file_path)


def process_pipeline(pipeline, source_image, mosaic_image, context): # Description read process, source_image
    print("process_pipeline = ", pipeline, source_image.shape, mosaic_image.shape)
    #context = {} # Empty dictionary
    for cmd in pipeline: # cmd = pipeline_call
        target_name = cmd["target"]
        operation_name = cmd["operation"]
        input_arguments = cmd.get("args", [])
        keyword_arguments_dict = {}
        for key in cmd:
            if key not in ["operation", "target", "args"]:
                keyword_arguments_dict[key] = convert_to_correct_type(cmd[key])

        # you get: ['B1', '=', 'equalized_greyscale', 'A1']
        args = dict(target_name=target_name, operation_name=operation_name, input_args=input_arguments, kw_args=keyword_arguments_dict, source_image=source_image, mosaic_image=mosaic_image, context=context)
        process_pipeline_instruction(args) # parts instead of pipeline


plugins = {}

def register(fun, name=None):
    name = name or fun.__name__
    plugins[name] = fun

def process_pipeline_instruction(args):
    operation_name = args["operation_name"]
    print("process_pipeline_instruction = ", args["target_name"], args["operation_name"], args["input_args"], args["source_image"].shape, args["mosaic_image"].shape)

    fun = plugins.get(operation_name)
    if not fun:
        raise Exception(f"Operation not available: {operation_name}")
    fun(args)


def process_pipeline_inst_src(args):
    target_name=args["target_name"]; input_arguments=args["input_args"]; source_image = args["source_image"]; mosaic_image=args["mosaic_image"]; context=args["context"]
    print("process_pipeline_inst_src = ", target_name, input_arguments, source_image.shape, mosaic_image.shape)
    x, y = name_to_coordinates(target_name, context) # Tuple unpacking from x, y to function
    print("name_to_coordinates = ", x, y)
    context[target_name] = source_image
    resized_image = resize(source_image, context["cell_width"], context["cell_height"])
    paste_image(resized_image, mosaic_image, x, y)


def put_image_to_mosaic(r, mosaic_image, target_name, context):
    """target name is name of the target cell "A0", context is the dictionary with image names and values as images"""
    x, y = name_to_coordinates(target_name, context) # Tuple unpacking from x, y to function
    print("name_to_coordinates = ", x, y)
    context[target_name] = r
    resized_image = resize(r, context["cell_width"], context["cell_height"])
    paste_image(resized_image, mosaic_image, x, y)


def process_pipeline_inst_greyscale(args): # target name is
    target_name=args["target_name"]; input_arguments=args["input_args"]; source_image = args["source_image"]; mosaic_image=args["mosaic_image"]; context=args["context"]
    print("target_name = ", target_name, "input_arguments = ", input_arguments, "source_image = ", source_image.shape, "mosaic_image = ", mosaic_image.shape)
    print("process_pipeline_inst_greyscale = ", target_name, input_arguments, source_image.shape, mosaic_image.shape)
    # Get the source image
    sources = [context[x] for x in input_arguments] # sources[0], sources[1]
    source_image = sources[0] # Will override the one in the parameters of the function
    r = source_image.copy() # Copy the image
    # set blue and green channels to 0,  0.299 ∙ Red + 0.587 ∙ Green + 0.114 ∙ Blue
    r[:, :, 0] = 0.299 * source_image[:,:,2] + 0.587 * source_image[:,:,1] + 0.114 * source_image[:,:,0]
    r[:, :, 1] = 0.299 * source_image[:,:,2] + 0.587 * source_image[:,:,1] + 0.114 * source_image[:,:,0]
    r[:, :, 2] = 0.299 * source_image[:,:,2] + 0.587 * source_image[:,:,1] + 0.114 * source_image[:,:,0]

    put_image_to_mosaic(r, mosaic_image, target_name, context)


def process_pipeline_inst_alpha_blend(args): # target name is
    target_name=args["target_name"]; input_arguments=args["input_args"]; source_image = args["source_image"]; mosaic_image=args["mosaic_image"]; context=args["context"]
    print("target_name = ", target_name, "input_arguments = ", input_arguments, "source_image = ", source_image.shape, "mosaic_image = ", mosaic_image.shape)
    print("process_pipeline_inst_alpha_blend = ", target_name, input_arguments, source_image.shape, mosaic_image.shape)
    # Get the source image
    sources = [context[x] for x in input_arguments] # sources[0], sources[1]
    source_image = sources[0] # Will override the one in the parameters of the function
    r = source_image.copy() # Copy the image
    # set blue and green channels to 0,  0.299 ∙ Red + 0.587 ∙ Green + 0.114 ∙ Blue
    r[:, :, 0] = 0.299 * source_image[:,:,2] + 0.587 * source_image[:,:,1] + 0.114 * source_image[:,:,0]
    r[:, :, 1] = 0.299 * source_image[:,:,2] + 0.587 * source_image[:,:,1] + 0.114 * source_image[:,:,0]
    r[:, :, 2] = 0.299 * source_image[:,:,2] + 0.587 * source_image[:,:,1] + 0.114 * source_image[:,:,0]

    put_image_to_mosaic(r, mosaic_image, target_name, context)


def process_pipeline_inst_red(args):
    target_name=args["target_name"]; input_arguments=args["input_args"]; source_image = args["source_image"]; mosaic_image=args["mosaic_image"]; context=args["context"]
    print("process_pipeline_inst_red = ", target_name, input_arguments, source_image.shape, mosaic_image.shape)
    # Get the source image
    sources = [context[x] for x in input_arguments] # sources[0], sources[1]
    source_image = sources[0] # Will override the one in the parameters of the function
    r = source_image.copy() # Copy the image
    # set blue and green channels to 0, blue is 0, green is 1, red is 2
    r[:, :, 0] = 0
    r[:, :, 1] = 0

    put_image_to_mosaic(r, mosaic_image, target_name, context)


def process_pipeline_inst_green(args):
    target_name=args["target_name"]; input_arguments=args["input_args"]; source_image = args["source_image"]; mosaic_image=args["mosaic_image"]; context=args["context"]
    print("process_pipeline_inst_green = ", target_name, input_arguments, source_image.shape, mosaic_image.shape)
    # Get the source image
    sources = [context[x] for x in input_arguments] # sources[0], sources[1]
    source_image = sources[0] # Will override the one in the parameters of the function
    r = source_image.copy() # Copy the image
    # set blue and  channels to 0
    r[:, :, 0] = 0
    r[:, :, 2] = 0

    put_image_to_mosaic(r, mosaic_image, target_name, context)


def process_pipeline_inst_blue(args):
    target_name=args["target_name"]; input_arguments=args["input_args"]; source_image = args["source_image"]; mosaic_image=args["mosaic_image"]; context=args["context"]
    print("process_pipeline_inst_blue = ", target_name, input_arguments, source_image.shape, mosaic_image.shape)
    # Get the source image
    sources = [context[x] for x in input_arguments] # sources[0], sources[1]
    source_image = sources[0] # Will override the one in the parameters of the function
    r = source_image.copy() # Copy the image
    # set blue and  channels to 0
    r[:, :, 1] = 0
    r[:, :, 2] = 0

    put_image_to_mosaic(r, mosaic_image, target_name, context)



def process_pipeline_inst_equalized(args):
    target_name=args["target_name"]; input_arguments=args["input_args"]; source_image = args["source_image"]; mosaic_image=args["mosaic_image"]; context=args["context"]
    print("process_pipeline_inst_equalized = ", target_name, input_arguments, source_image.shape, mosaic_image.shape)
    # Get the source image
    sources = [context[x] for x in input_arguments] # sources[0], sources[1]
    source_image = sources[0] # Will override the one in the parameters of the function
    r = equalize_histogram_rgb(source_image)

    put_image_to_mosaic(r, mosaic_image, target_name, context)


def process_pipeline_inst_edge_canny(args):
    target_name=args["target_name"]; input_arguments=args["input_args"]; source_image = args["source_image"]; mosaic_image=args["mosaic_image"]; context=args["context"]
    print("process_pipeline_inst_edge_canny = ", target_name, input_arguments, source_image.shape, mosaic_image.shape)

    x, y = name_to_coordinates(target_name, context) # Tuple unpacking from x, y to function
    print("name_to_coordinates = ", x, y)
    sources = [context[x] for x in input_arguments]  # sources[0], sources[1]
    source_image = context[sources[0]]  # Will override the one in the parameters of the function
    r = canny_edge_detect(sources[0])
    put_image_to_mosaic(r, mosaic_image, target_name, context)


def process_pipeline_inst_edge_detect(args):
    target_name=args["target_name"]; input_arguments=args["input_args"]; source_image = args["source_image"]; mosaic_image=args["mosaic_image"]; context=args["context"]
    print("process_pipeline_inst_edge_detect = ", target_name, input_arguments, source_image.shape, source_image.dtype)

    sources = [context[x] for x in input_arguments] # sources[0], sources[1]
    source_image = sources[0]  # Will override the one in the parameters of the function

    img_gray = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)

    try:
        print("img_gray.shape=", img_gray.shape, img_gray.dtype)
        segments = lsd(img_gray, scale=0.5)
        print("Segments:", segments)
        showimage(img_gray, cmap="gray")

        img2 = source_image.copy()
        draw_lines_to_image(img2, segments, min_length=50)
        showimage(img2)

        r = img2
        put_image_to_mosaic(r, mosaic_image, target_name, context)

    except Exception as e:
        print("process_pipeline_inst_edge_detect: Exception occurred!", e)


def process_pipeline_inst_gaussian_blur(args):
    target_name=args["target_name"]; input_arguments=args["input_args"]; source_image = args["source_image"]; mosaic_image=args["mosaic_image"]; context=args["context"]
    print("process_pipeline_inst_gaussian_blur = ", target_name, input_arguments, source_image.shape, mosaic_image.shape)
    sources = [context[x] for x in input_arguments] # sources[0], sources[1]
    source_image = sources[0] # Will override the one in the parameters of the function
    r = gaussian_blur(source_image)
    put_image_to_mosaic(r, mosaic_image, target_name, context)


def process_pipeline_inst_sobel(args):
    target_name=args["target_name"]; input_arguments=args["input_args"]; source_image = args["source_image"]; mosaic_image=args["mosaic_image"]; context=args["context"]
    print("process_pipeline_inst_sobel = ", target_name, input_arguments, source_image.shape, mosaic_image.shape)

    sources = [context[x] for x in input_arguments] # sources[0], sources[1]
    source_image = sources[0] # Will override the one in the parameters of the function
    r = gray2rgb(sobel(grayscale(source_image)))
    put_image_to_mosaic(r, mosaic_image, target_name, context)


def process_pipeline_inst_threshold(args):
    target_name=args["target_name"]; input_arguments=args["input_args"]; source_image = args["source_image"]; mosaic_image=args["mosaic_image"]; context=args["context"]; keyword_args=args["kw_args"]
    print("process_pipeline_inst_threshold = ", target_name, input_arguments, source_image.shape, mosaic_image.shape)
    sources = [context[x] for x in input_arguments] # sources[0], sources[1]
    grayscaled_image = grayscale(sources[0])
    rgb_image = cv2.cvtColor(grayscaled_image, cv2.COLOR_GRAY2RGB)
    r = threshold(rgb_image, thresh=keyword_args["lower"], maxval=keyword_args["upper"])
    put_image_to_mosaic(r, mosaic_image, target_name, context)


def process_pipeline_inst_dilate(args):
    target_name=args["target_name"]; input_arguments=args["input_args"]; source_image = args["source_image"]; mosaic_image=args["mosaic_image"]; context=args["context"]; keyword_args=args["kw_args"]
    print("process_pipeline_inst_dilate = ", target_name, input_arguments, source_image.shape, mosaic_image.shape)
    kernel_size = int(keyword_args.get("size", 1))
    sources = [context[x] for x in input_arguments] # sources[0], sources[1]
    r = dilate(sources[0], kernel_size=(kernel_size, kernel_size))
    put_image_to_mosaic(r, mosaic_image, target_name, context)


def process_pipeline_inst_erode(args):
    target_name=args["target_name"]; input_arguments=args["input_args"]; source_image = args["source_image"]; mosaic_image=args["mosaic_image"]; context=args["context"]; keyword_args=args["kw_args"]
    print("process_pipeline_inst_dilate = ", target_name, input_arguments, source_image.shape, mosaic_image.shape)
    kernel_size = int(keyword_args.get("size", 1))
    sources = [context[x] for x in input_arguments] # sources[0], sources[1]
    r = erode(sources[0], kernel_size=(kernel_size, kernel_size))
    put_image_to_mosaic(r, mosaic_image, target_name, context)


def process_pipeline_inst_xdog(args):
    target_name=args["target_name"]; input_arguments=args["input_args"]; source_image = args["source_image"]; mosaic_image=args["mosaic_image"]; context=args["context"]
    print("process_pipeline_inst_xdog = ", target_name, input_arguments, source_image.shape, mosaic_image.shape)
    sources = [context[x] for x in input_arguments] # sources[0]
    grayscaled_image = grayscale(sources[0])
    r = xdog(grayscaled_image)
    # put_image_to_mosaic(r, mosaic_image, target_name, context)


def process_pipeline_inst_alpha_blending(args):
    target_name=args["target_name"]; input_arguments=args["input_args"]; source_image = args["source_image"]; mosaic_image=args["mosaic_image"]; context=args["context"]; keyword_args=args["kw_args"]
    print("process_pipeline_inst_alpha_blending = ", target_name, input_arguments, source_image.shape, mosaic_image.shape)

    sources = [context[x] for x in input_arguments] # sources[0], sources[1]
    r = alpha_blending(sources[0], sources[1], alpha=keyword_args["alpha"])
    put_image_to_mosaic(r, mosaic_image, target_name, context)


def process_pipeline_inst_laplace(args):
    target_name=args["target_name"]; input_arguments=args["input_args"]; source_image = args["source_image"]; mosaic_image=args["mosaic_image"]; context=args["context"]; keyword_args=args["kw_args"]
    print("process_pipeline_inst_laplace = ", target_name, input_arguments, source_image.shape, mosaic_image.shape)

    sources = [context[x] for x in input_arguments] # sources[0], sources[1]
    r = laplace(sources[0])
    put_image_to_mosaic(r, mosaic_image, target_name, context)


def process_pipeline_inst_load_image(args):
    target_name=args["target_name"]; input_arguments=args["input_args"]; source_image = args["source_image"]; mosaic_image=args["mosaic_image"]; context=args["context"]; keyword_args=args["kw_args"]
    print("process_pipeline_inst_xdog = ", target_name, input_arguments, source_image.shape, mosaic_image.shape)

    r = cv2.imread(keyword_args["path"])
    put_image_to_mosaic(r, mosaic_image, target_name, context)


def process_pipeline_inst_binary_and(args):
    target_name=args["target_name"]; input_arguments=args["input_args"]; source_image = args["source_image"]; mosaic_image=args["mosaic_image"]; context=args["context"]; keyword_args=args["kw_args"]
    print("process_pipeline_inst_and = ", target_name, input_arguments, source_image.shape, mosaic_image.shape)

    sources = [context[x] for x in input_arguments] # sources[0], sources[1]
    r = binary_and(sources[0], sources[1])
    put_image_to_mosaic(r, mosaic_image, target_name, context)


def process_pipeline_inst_binary_xor(args):
    target_name=args["target_name"]; input_arguments=args["input_args"]; source_image = args["source_image"]; mosaic_image=args["mosaic_image"]; context=args["context"]
    print("process_pipeline_inst_xor = ", target_name, input_arguments, source_image.shape, mosaic_image.shape)

    sources = [context[x] for x in input_arguments] # sources[0], sources[1]
    r = binary_xor(sources[0], sources[1])
    put_image_to_mosaic(r, mosaic_image, target_name, context)


def process_pipeline_inst_outline(args):
    target_name=args["target_name"]; input_arguments=args["input_args"]; source_image = args["source_image"]; mosaic_image=args["mosaic_image"]; context=args["context"]
    print("process_pipeline_inst_xdog = ", target_name, input_arguments, source_image.shape, mosaic_image.shape)
    sources = [context[x] for x in input_arguments] # sources[0]
    r = outline(sources[0])
    put_image_to_mosaic(r, mosaic_image, target_name, context)


def process_pipeline_inst_colorize_mask(args):
    target_name=args["target_name"]; input_arguments=args["input_args"]; source_image = args["source_image"]; mosaic_image=args["mosaic_image"]; context=args["context"]; keyword_args=args["kw_args"]
    print("process_pipeline_inst_colorize_mask = ", target_name, input_arguments, source_image.shape, mosaic_image.shape)
    sources = [context[x] for x in input_arguments] # sources[0]
    converted_color = convert_hash_to_bgr_color(keyword_args["color"])
    r = colorize_mask(sources[0], color=converted_color)
    put_image_to_mosaic(r, mosaic_image, target_name, context)


def process_pipeline_inst_color_to_mask(args):
    target_name=args["target_name"]; input_arguments=args["input_args"]; source_image = args["source_image"]; mosaic_image=args["mosaic_image"]; context=args["context"]; keyword_args=args["kw_args"]
    print("process_pipeline_inst_colorize_mask = ", target_name, input_arguments, source_image.shape, mosaic_image.shape)
    sources = [context[x] for x in input_arguments] # sources[0]
    converted_color = convert_hash_to_bgr_color(keyword_args["color"])
    r = color_to_mask(sources[0], color=converted_color)
    put_image_to_mosaic(r, mosaic_image, target_name, context)


def process_pipeline_inst_(args):
    target_name=args["target_name"]; input_arguments=args["input_args"]; source_image = args["source_image"]; mosaic_image=args["mosaic_image"]; context=args["context"]; keyword_args=args["kw_args"]
    print("process_pipeline_inst_alpha_blending = ", target_name, input_arguments, source_image.shape, mosaic_image.shape)

    sources = [context[x] for x in input_arguments] # sources[0], sources[1]
    r = alpha_blending(sources[0], sources[1], alpha=keyword_args["alpha"])
    put_image_to_mosaic(r, mosaic_image, target_name, context)


def name_to_coordinates(name, context): # Description: Transforms a name to coordinates
    #print(name, context)
    letter = name[0] # Holds "n"
    number = name[1] # Holds "1" from n1
    row = ord(letter) - ord('A') # ord = order of letter, for 'A' it returns ASCII
    col = ord(number) - ord('0') #
    x = col * context["cell_width"] # Change the index to col * cell_width
    y = row * context["cell_height"]
    return x, y


# register plugin functions
# Replaces the if cycle
register(process_pipeline_inst_src, "src")
register(process_pipeline_inst_greyscale, "greyscale")
register(process_pipeline_inst_red, "red")
register(process_pipeline_inst_green, "green")
register(process_pipeline_inst_blue, "blue")
register(process_pipeline_inst_equalized, "equalized")
register(process_pipeline_inst_edge_canny, "edge_canny")
register(process_pipeline_inst_edge_detect, "edge_detect_LSD")
register(process_pipeline_inst_erode, "erode")
register(process_pipeline_inst_gaussian_blur, "gaussian")
register(process_pipeline_inst_sobel, "sobel")
register(process_pipeline_inst_threshold, "threshold")
register(process_pipeline_inst_laplace, "laplace")
register(process_pipeline_inst_dilate, "dilate")
register(process_pipeline_inst_xdog, "xdog")
register(process_pipeline_inst_alpha_blending, "alpha_blend")
register(process_pipeline_inst_load_image, "load_image")
register(process_pipeline_inst_binary_and, "binary_and")
register(process_pipeline_inst_binary_xor, "binary_xor")
register(process_pipeline_inst_outline, "outline")
register(process_pipeline_inst_colorize_mask, "colorize_mask")
register(process_pipeline_inst_color_to_mask, "color_to_mask")
