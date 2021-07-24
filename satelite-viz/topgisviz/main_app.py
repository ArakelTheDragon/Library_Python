#!/usr/bin/env python

# simple.py

import wx
from ciirc_utils import alpha_blending, showimage, reading_in_an_image, xdog, dog, hatch, threshold, sobel, laplace, gaussian_blur, erode, edge_canny, grayscale, save_to_file
import sys, os, glob
import cv2
import numpy as np

# Global variables
dstpath = 'target/'

# Function definitions
def onButton_test(event):
    print ("Button pressed.")


def create_text_input(event):
    # Create text input
    dlg = wx.TextEntryDialog(frame, 'Enter some text', 'Text Entry')  # Set the parameters of the text field
    dlg.SetValue("images/")  # Set the default text in the text field
    if dlg.ShowModal() == wx.ID_OK:
        print('You entered: %s\n' % dlg.GetValue())
        string_value = dlg.GetValue() # Get the value of the text box
    print("create_text_input finished")
    dlg.Destroy()


def button_alpha_function(event):
    # Create text input
    dlg = wx.TextEntryDialog(frame, 'Enter some text', 'Text Entry')  # Set the parameters of the text field
    dlg.SetValue("images/")  # Set the default text in the text field
    if dlg.ShowModal() == wx.ID_OK:
        string_value = dlg.GetValue() # Get the value of the text box
        files = os.listdir(string_value)                         # Read the files from source_directory and record them in a list
        print(sorted(files))
        for image in files:                                 # For index in list
            print("Enter the for.")
            img_alpha = cv2.imread(os.path.join(string_value, image))     # Read the image from path + image name as an array into img
            img_backdroung = cv2.imread(os.path.join(string_value, image+1))     # Read the image from path + image name as an array into img
            img_foreground = cv2.imread(os.path.join(string_value, image+2))     # Read the image from path + image name as an array into img
            #grayscaled_image = grayscale(img)
            alphaed_image = alpha_blending(img_alpha, img_backdroung, img_backdroung)  # grayscale every image in the list input_images
            cv2.imwrite(os.path.join(dstpath, image), alphaed_image) # Write the image gray to a file with name
            print("Image written =", image)
    print("button_alpha_function finished")
    dlg.Destroy()


def button_xdog_function(event):
    print("Enter the button_xdog_function.")
    # Create text input
    dlg = wx.TextEntryDialog(frame, 'Enter some text', 'Text Input')  # Set the parameters of the text field
    dlg.SetValue("images/")  # Set the default text in the text field
    print("Before the if.")
    if dlg.ShowModal() == wx.ID_OK:
        string_value = dlg.GetValue() # Get the value of the text box
        files = os.listdir(string_value)                         # Read the files from source_directory and record them in a list
        print(sorted(files))
        for image in files:                                 # For index in list
            print("Enter the for.")
            img = cv2.imread(os.path.join(string_value, image))     # Read the image from path + image name as an array into img
            grayscaled_image = grayscale(img)
            xdoged_image = xdog(grayscaled_image)  # grayscale every image in the list input_images
            cv2.imwrite(os.path.join(dstpath, image), xdoged_image) # Write the image gray to a file with name
            print("Image written =", image)
    print("button_xdog_function finished")
    dlg.Destroy()


def button_threshold_function(event):
    # Create text input
    dlg = wx.TextEntryDialog(frame, 'Enter some text', 'Text Entry')  # Set the parameters of the text field
    dlg.SetValue("images/")  # Set the default text in the text field
    if dlg.ShowModal() == wx.ID_OK:
        string_value = dlg.GetValue() # Get the value of the text box
        files = os.listdir(string_value)                         # Read the files from source_directory and record them in a list
        print(sorted(files))
        for image in files:                                 # For index in list
            print("Enter the for.")
            img = cv2.imread(os.path.join(string_value, image))     # Read the image from path + image name as an array into img
            grayscaled_image = grayscale(img)
            thresholded_image = threshold(grayscaled_image)  # grayscale every image in the list input_images
            cv2.imwrite(os.path.join(dstpath, image), thresholded_image) # Write the image gray to a file with name
            print("Image written =", image)
    print("button_threshold_function finished")
    dlg.Destroy()


def button_sobel_function(event):
    # Create text input
    dlg = wx.TextEntryDialog(frame, 'Enter some text', 'Text Entry')  # Set the parameters of the text field
    dlg.SetValue("images/")  # Set the default text in the text field
    if dlg.ShowModal() == wx.ID_OK:
        string_value = dlg.GetValue() # Get the value of the text box
        files = os.listdir(string_value)                         # Read the files from source_directory and record them in a list
        print(sorted(files))
        for image in files:                                 # For index in list
            print("Enter the for.")
            img = cv2.imread(os.path.join(string_value, image))     # Read the image from path + image name as an array into img
            grayscaled_image = grayscale(img)
            sobeled_image = sobel(grayscaled_image)  # grayscale every image in the list input_images
            cv2.imwrite(os.path.join(dstpath, image), sobeled_image) # Write the image gray to a file with name
            print("Image written =", image)
    print("button_sobel_function finished")
    dlg.Destroy()


def button_laplace_function(event):
    # Create text input
    dlg = wx.TextEntryDialog(frame, 'Enter some text', 'Text Entry')  # Set the parameters of the text field
    dlg.SetValue("images/")  # Set the default text in the text field
    if dlg.ShowModal() == wx.ID_OK:
        string_value = dlg.GetValue() # Get the value of the text box
        files = os.listdir(string_value)                         # Read the files from source_directory and record them in a list
        print(sorted(files))
        for image in files:                                 # For index in list
            print("Enter the for.")
            img = cv2.imread(os.path.join(string_value, image))     # Read the image from path + image name as an array into img
            grayscaled_image = grayscale(img)
            laplaced_image = laplace(grayscaled_image)  # grayscale every image in the list input_images
            cv2.imwrite(os.path.join(dstpath, image), laplaced_image) # Write the image gray to a file with name
            print("Image written =", image)
    print("button_laplace_function finished")
    dlg.Destroy()


def button_gaussian_blur_function(event):
    # Create text input
    dlg = wx.TextEntryDialog(frame, 'Enter some text', 'Text Entry')  # Set the parameters of the text field
    dlg.SetValue("images/")  # Set the default text in the text field
    if dlg.ShowModal() == wx.ID_OK:
        string_value = dlg.GetValue() # Get the value of the text box
        files = os.listdir(string_value)                         # Read the files from source_directory and record them in a list
        print(sorted(files))
        for image in files:                                 # For index in list
            print("Enter the for.")
            img = cv2.imread(os.path.join(string_value, image))     # Read the image from path + image name as an array into img
            grayscaled_image = grayscale(img)
            gaussian_blured_image = gaussian_blur(grayscaled_image)  # grayscale every image in the list input_images
            cv2.imwrite(os.path.join(dstpath, image), gaussian_blured_image) # Write the image gray to a file with name
            print("Image written =", image)
    print("button_gaussian_blur_function finished")
    dlg.Destroy()


def button_edge_canny_function(event):
    # Create text input
    dlg = wx.TextEntryDialog(frame, 'Enter some text', 'Text Entry')  # Set the parameters of the text field
    dlg.SetValue("images/")  # Set the default text in the text field
    if dlg.ShowModal() == wx.ID_OK:
        string_value = dlg.GetValue() # Get the value of the text box
        files = os.listdir(string_value)                         # Read the files from source_directory and record them in a list
        print(sorted(files))
        for image in files:                                 # For index in list
            print("Enter the for.")
            img = cv2.imread(os.path.join(string_value, image))     # Read the image from path + image name as an array into img
            grayscaled_image = grayscale(img)
            edged_image = edge_canny(grayscaled_image)  # grayscale every image in the list input_images
            cv2.imwrite(os.path.join(dstpath, image), edged_image) # Write the image gray to a file with name
            print("Image written =", image)
    print("button_edge_canny_function finished")
    dlg.Destroy()


def button_erode_function(event):
    # Create text input
    dlg = wx.TextEntryDialog(frame, 'Enter some text', 'Text Entry')  # Set the parameters of the text field
    dlg.SetValue("images/")  # Set the default text in the text field
    if dlg.ShowModal() == wx.ID_OK:
        string_value = dlg.GetValue() # Get the value of the text box
        files = os.listdir(string_value)                         # Read the files from source_directory and record them in a list
        print(sorted(files))
        for image in files:                                 # For index in list
            print("Enter the for.")
            img = cv2.imread(os.path.join(string_value, image))     # Read the image from path + image name as an array into img
            grayscaled_image = grayscale(img)
            eroded_image = erode(grayscaled_image)  # grayscale every image in the list input_images
            cv2.imwrite(os.path.join(dstpath, image), eroded_image) # Write the image gray to a file with name
            print("Image written =", image)
    print("button_erode_function finished")
    dlg.Destroy()


def button_grayscale_function(event):
    # Create text input
    dlg = wx.TextEntryDialog(frame, 'Enter some text', 'Text Entry')  # Set the parameters of the text field
    dlg.SetValue("images/")  # Set the default text in the text field
    if dlg.ShowModal() == wx.ID_OK:
        string_value = dlg.GetValue() # Get the value of the text box
        files = os.listdir(string_value)                         # Read the files from source_directory and record them in a list
        print(sorted(files))
        for image in files:                                 # For index in list
            print("Enter the for.")
            img = cv2.imread(os.path.join(string_value, image))     # Read the image from path + image name as an array into img
            grayscaled_image = grayscale(img)
            cv2.imwrite(os.path.join(dstpath, image), grayscaled_image) # Write the image gray to a file with name
            print("Image written =", image)
    print("button_grayscale_function finished")
    dlg.Destroy()


# The beginning of the application
app = wx.App()

# The frame of the application
frame = wx.Frame(parent=None, title='Simple application')
frame.SetDimensions(0, 0, 2000, 500) # 0, 0 are the start coordinates, 2000, 500 are the dimensions

# Panel containing all buttons
panel = wx.Panel(parent=frame, id=wx.ID_ANY)
button_test = wx.Button(parent=panel, id=wx.ID_ANY, label='Test', pos=(10, 10), size=(130, 30), style=0, name="Button_test")
button_test.Bind(wx.EVT_BUTTON, onButton_test)
button_alpha = wx.Button(parent=panel, id=wx.ID_ANY, label='button_alpha', pos=(140, 10), size=(130, 30), style=0, name="button_alpha")
button_alpha.Bind(wx.EVT_BUTTON, button_alpha_function)
button_xdog = wx.Button(parent=panel, id=wx.ID_ANY, label='button_xdog', pos=(270, 10), size=(130, 30), style=0, name="button_xdog")
button_xdog.Bind(wx.EVT_BUTTON, button_xdog_function)
button_threshold = wx.Button(parent=panel, id=wx.ID_ANY, label='button_threshold', pos=(400, 10), size=(130, 30), style=0, name="button_threshold")
button_threshold.Bind(wx.EVT_BUTTON, button_threshold_function)
button_sobel = wx.Button(parent=panel, id=wx.ID_ANY, label='button_sobel', pos=(530, 10), size=(130, 30), style=0, name="button_sobel")
button_sobel.Bind(wx.EVT_BUTTON, button_sobel_function)
button_laplace = wx.Button(parent=panel, id=wx.ID_ANY, label='button_laplace', pos=(660, 10), size=(130, 30), style=0, name="button_laplace")
button_laplace.Bind(wx.EVT_BUTTON, button_laplace_function)
button_gaussian_blur = wx.Button(parent=panel, id=wx.ID_ANY, label='button_gaussian', pos=(790, 10), size=(130, 30), style=0, name="button_gaussian_blur")
button_gaussian_blur.Bind(wx.EVT_BUTTON, button_gaussian_blur_function)
button_erode = wx.Button(parent=panel, id=wx.ID_ANY, label='button_erode', pos=(920, 10), size=(130, 30), style=0, name="button_erode")
button_erode.Bind(wx.EVT_BUTTON, button_erode_function)
button_edge_canny = wx.Button(parent=panel, id=wx.ID_ANY, label='button_edge', pos=(1050, 10), size=(130, 30), style=0, name="button_edge_canny")
button_edge_canny.Bind(wx.EVT_BUTTON, button_edge_canny_function)
button_grayscale = wx.Button(parent=panel, id=wx.ID_ANY, label='button_grayscale', pos=(1180, 10), size=(130, 30), style=0, name="button_grayscale")
button_grayscale.Bind(wx.EVT_BUTTON, button_grayscale_function)

frame.Show()


app.MainLoop()