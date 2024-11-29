"""Include your imports here
Some example imports are below"""

import numpy as np 
from skimage import io, img_as_ubyte,exposure
import matplotlib.pyplot as plt
import math

def part1_histogram_compute():
    filename = r'test.jpg'
    image = io.imread(filename, as_gray=True)
    img = img_as_ubyte(image)

    n = 64
    hist = np.zeros([n], dtype=int)
    # Histogram computed by your code (cannot use in-built functions!)
    for row in img:
        for pixel_value in row:
            bin_index = int(pixel_value * n / 256)
            hist[bin_index] += 1
    hist_np, _ = np.histogram(img.flatten(), bins=n, range=[0, 256]) # Histogram computed by numpy

    """Plotting code provided here
    Make sure to match the variable names in your code!"""

    plt.figure(figsize=(8, 10))
    plt.subplot(121), plt.plot(hist), plt.title('My Histogram')
    plt.xlim([0, n])
    plt.subplot(122), plt.plot(hist_np), plt.title('Numpy Histogram')
    plt.xlim([0, n])

    plt.show()

def part2_histogram_equalization():
    filename = r'test.jpg'
    image = io.imread(filename, as_gray=True)
    img = img_as_ubyte(image)

 
    """add your code here"""
    n_bins = 64

    # 64-bin Histogram computed by your code (cannot use in-built functions!)
    hist = np.zeros(n_bins, dtype=int)

    ## HINT: Initialize another image (you can use np.zeros) and update the pixel intensities in every location
    for row in img:
        for pixel_value in row:
            bin_index = int(pixel_value * n_bins / 256)
            hist[bin_index] += 1

    img_eq1 = np.zeros_like(img) 
    cumulative_sum = np.cumsum(hist)


    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pixel_value = img[i, j]
            a= math.floor(pixel_value/255*n_bins)
            img_eq1[i, j] = math.floor(255/(img.shape[0]*img.shape[1])*cumulative_sum[a]+0.5)

    # Histogram of equalized image
    hist_eq = np.zeros(n_bins, dtype=int)
    for row in img_eq1:
        for pixel_value in row:
            bin_index = int(pixel_value * n_bins / 256)
            hist_eq[bin_index] += 1

    """Plotting code provided here
    Make sure to match the variable names in your code!"""

    plt.figure(figsize=(8, 10))
    plt.subplot(221), plt.imshow(image, 'gray'), plt.title('Original Image')
    plt.subplot(222), plt.plot(hist), plt.title('Histogram')
    plt.xlim([0, n_bins])
    plt.subplot(223), plt.imshow(img_eq1, 'gray'), plt.title('New Image')
    plt.subplot(224), plt.plot(hist_eq), plt.title('Histogram After Equalization')
    plt.xlim([0, n_bins])
    
    plt.show() 

def part3_histogram_comparing():

    filename1 = 'day.jpg'
    filename2 = 'night.jpg'

    # Read in the image
    img1 = io.imread(filename1, as_gray=True)
    # Read in another image
    img2 = io.imread(filename2, as_gray=True)

    # Calculate the histograms for img1 and img2 using skimage.exposure.histogram
    hist1, _ = exposure.histogram(img1, nbins=256)
    hist2, _ = exposure.histogram(img2, nbins=256)

    # Normalize the histograms for img1 and img2
    hist1_norm = hist1 / np.sum(hist1)
    hist2_norm = hist2 / np.sum(hist2)

    # Calculate the Bhattacharya coefficient
    # Value must be close to 0.87
    bc = np.sum(np.sqrt(np.multiply(hist1_norm , hist2_norm)))
    print("Bhattacharyya Coefficient: ", bc)

def part4_histogram_matching():
    filename1 = 'day.jpg'
    filename2 = 'night.jpg'

    #============Grayscale============

    # Read in the image
    source_gs = io.imread(filename1,
                           as_gray=True
                           )
    source_gs = img_as_ubyte(source_gs)
    # Read in another image
    template_gs = io.imread(filename2,
                             as_gray=True
                             )
    template_gs = img_as_ubyte(template_gs)
    
    
    """add your code here"""

    # Compute histograms using numpy
    hist_source, _ = np.histogram(source_gs.flatten(), bins=256, range=[0, 256])
    hist_template, _ = np.histogram(template_gs.flatten(), bins=256, range=[0, 256])

    # Compute cumulative distribution functions (CDFs)
    cdf_source = np.cumsum(hist_source) / np.sum(hist_source)
    cdf_template = np.cumsum(hist_template) / np.sum(hist_template)

    A = np.zeros(256, dtype=int)
    for a in range(256):
        j = 0
        while cdf_source[a] > cdf_template[j]:
            j += 1
        A[a] = j

    matched_gs = np.zeros_like(source_gs)
    
    for i in range(source_gs.shape[0]):
        for j in range(source_gs.shape[1]):
            a = source_gs[i, j]
            matched_gs[i, j] = A[a]


    fig = plt.figure()
    gs = plt.GridSpec(1, 3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharex=ax1, sharey=ax1)
    ax3 = fig.add_subplot(gs[0, 2], sharex=ax1, sharey=ax1)

    for aa in (ax1, ax2, ax3):
        aa.set_axis_off()

    ax1.imshow(source_gs, cmap=plt.cm.gray)
    ax1.set_title('source_gs')
    ax2.imshow(template_gs, cmap=plt.cm.gray)
    ax2.set_title('template_gs')
    ax3.imshow(matched_gs, cmap=plt.cm.gray)
    ax3.set_title('matched_gs')
    plt.show()


    #============RGB============
    # Read in the image
    source_rgb = io.imread(filename1,
                           # as_gray=True
                           )
    # Read in another image
    template_rgb = io.imread(filename2,
                             # as_gray=True
                             )
    
    """add your code here"""
    matched_rgb = np.zeros_like(source_rgb)
    
    for channel in range(3):
        hist_source, _ = np.histogram(source_rgb[:, :, channel].flatten(), bins=256, range=[0, 256])
        hist_template, _ = np.histogram(template_rgb[:, :, channel].flatten(), bins=256, range=[0, 256])

        cdf_source = np.cumsum(hist_source) / np.sum(hist_source)
        cdf_template = np.cumsum(hist_template) / np.sum(hist_template)

        A = np.zeros(256, dtype=int)
        for a in range(256):
            j = 0
            while cdf_source[a] > cdf_template[j]:
                j += 1
            A[a] = j

        matched_rgb[:, :, channel] = A[source_rgb[:, :, channel]]

    fig = plt.figure()
    gs = plt.GridSpec(1, 3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharex=ax1, sharey=ax1)
    ax3 = fig.add_subplot(gs[0, 2], sharex=ax1, sharey=ax1)

    for aa in (ax1, ax2, ax3):
        aa.set_axis_off()

    ax1.imshow(source_rgb)
    ax1.set_title('source_rgb')
    ax2.imshow(template_rgb)
    ax2.set_title('template_rgb')
    ax3.imshow(matched_rgb)
    ax3.set_title('matched_rgb')
    plt.show()
    
if __name__ == '__main__':
    part1_histogram_compute()
    part2_histogram_equalization()
    part3_histogram_comparing()
    part4_histogram_matching()

