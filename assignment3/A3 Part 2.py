import os
from sklearn.cluster import KMeans
from scipy import spatial
from skimage import io, color, img_as_float
import numpy as np
import matplotlib.pyplot as plt
from math import floor

# Finds the closest colour in thepip  palette using kd-tree.
def nearest(palette, colour):
    dist, i = palette.query(colour)
    return palette.data[i]


# Make a kd-tree palette from the provided list of colours
def makePalette(colours):
    return spatial.KDTree(colours)


# Dynamically calculates an N-colour palette for the given image
# Uses the KMeans clustering algorithm to determine the best colours
# Returns a kd-tree palette with those colours
def findPalette(image, nColours):
    # TODO: perform KMeans clustering to get 'colours' --  the computed k means
    reshaped_image = image.reshape((-1, 3))  # Reshape the image to 2D array
    kmeans = KMeans(n_clusters=nColours, random_state=42)
    kmeans.fit(reshaped_image)
    colours = kmeans.cluster_centers_

    colours_img = np.zeros((50, int(nColours*50), 3), dtype=np.float32)
    start_id = 0
    for col_id in range(nColours):
        end_id = start_id + 50
        colours_img[:, start_id:end_id, :] = colours[col_id, :]
        start_id = end_id

    print(f'colours:\n{colours}')

    plt.figure(figsize=(10, 5))
    plt.imshow(colours_img)

    return makePalette(colours)

'''
def ModifiedFloydSteinbergDitherColor(image, palette):

    The following pseudo-code for a grayscale image is grabbed from Wikipedia:
    https://en.wikipedia.org/wiki/Floyd%E2%80%93Steinberg_dithering.
    The error distribution has been modified according to the instructions on eClass.

    total_abs_err := 0
    for each y from top to bottom ==> (height)
        for each x from left to right ==> (width)
            oldpixel  := image[x][y]
            newpixel  := nearest(oldpixel) # Determine the new colour for the current pixel from palette
            image[x][y]  := newpixel
            quant_error  := oldpixel - newpixel

            total_abs_err := total_abs_err + abs(quant_error)

            image[x + 1][y    ] := image[x + 1][y    ] + quant_error * 11 / 26
            image[x - 1][y + 1] := image[x - 1][y + 1] + quant_error * 5 / 26
            image[x    ][y + 1] := image[x    ][y + 1] + quant_error * 7 / 26
            image[x + 1][y + 1] := image[x + 1][y + 1] + quant_error * 3 / 26

    avg_abs_err := total_abs_err / image.size
    """

    # TODO: implement agorithm for RGB image (hint: you need to handle error in each channel separately)

    return image
'''

def ModifiedFloydSteinbergDitherColor(image, palette):
    height, width, _ = image.shape
    total_abs_err = 0

    for y in range(height):
        for x in range(width):
            old_pixel = image[y, x, :]

            # Determine the new colour for the current pixel from palette
            new_pixel = nearest(palette, old_pixel)
            image[y, x, :] = new_pixel

            # Quantization error
            quant_error = old_pixel - new_pixel
            total_abs_err += np.sum(np.abs(quant_error))

            # Diffusion of error to neighboring pixels
            if x + 1 < width:
                image[y, x + 1, :] += quant_error * 11 / 26
            if x - 1 >= 0 and y + 1 < height:
                image[y + 1, x - 1, :] += quant_error * 5 / 26
            if y + 1 < height:
                image[y + 1, x, :] += quant_error * 7 / 26
            if x + 1 < width and y + 1 < height:
                image[y + 1, x + 1, :] += quant_error * 3 / 26

    avg_abs_err = total_abs_err / (height * width * 3)  # Average error per channel per pixel

    return image

if __name__ == "__main__":
    # The number colours: change to generate a dynamic palette
    nColours = 7

    # read image
    imfile = "mandrill.png"
    image = io.imread(imfile)
    orig = image.copy()

    # Strip the alpha channel if it exists
    image = image[:, :, :3]

    # Convert the image from 8bits per channel to floats in each channel for precision
    image = img_as_float(image)

    # Dynamically generate an N colour palette for the given image
    palette = findPalette(image, nColours)
    colours = palette.data
    colours = img_as_float([colours.astype(np.ubyte)])[0]

    # call dithering function
    img = ModifiedFloydSteinbergDitherColor(image, palette)

    # show
    plt.figure(figsize=(10, 5))
    plt.subplot(121), plt.imshow(orig), plt.title('Original Image')
    plt.subplot(122), plt.imshow(img), plt.title(f'Dithered Image (nColours = {nColours})')
    plt.show()
