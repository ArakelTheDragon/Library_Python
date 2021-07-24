# Library includes
from ciir_utils import reading_in_an_image, showimage


# Function definitions
def colorize_mask(img, color=[255, 0, 0]): # [255, 0, 0] = red in RGB format
    height, width, _ = img.shape

    for i in range(height):
        for j in range(width):
            # img[i,j] is the RGB pixel at position (i, j)
            # check if it's [0, 0, 0] and replace with [255, 255, 255] if so
            if img[i,j].sum() == 0:
                img[i, j] = color
    return img


# Function calls
read_image = cv2.imread("images/utm_00000.mask.png")
colorised_image = colorize_mask(read_image)
cv2.imshow(colorised_image)