# image_processing
This repo includes the projects related to image processing in 206 class. Thanks for all TAs and professor Nilanjan Ray's help.

# Assignment 1: Histogram-Based Image Processing

This assignment focuses on implementing and exploring various histogram-based techniques in image processing, including:

1. **Histogram Computation**: Manually calculating histograms to understand intensity distribution in images.  
2. **Histogram Equalization**: Enhancing image contrast by redistributing pixel intensities.  
3. **Histogram Comparison**: Measuring similarity between image histograms using the Bhattacharyya coefficient.  
4. **Histogram Matching**: Transforming an image's histogram to match another's for visual consistency.

These techniques provide foundational tools for analyzing and enhancing images effectively.

# Assignment 2: Image Filtering and Edge Detection

This assignment implements a range of image processing techniques to explore filtering, enhancement, restoration, and edge detection, including:

1. **Image Filtering**:  
   - Applied Laplacian, Gaussian, and custom kernels to filter images and observe effects on image details and noise reduction.  
   - Enhanced images using Laplacian and Gaussian filters combined with sharpening techniques (e.g., unsharp masking).  

2. **Noise Removal**:  
   - Used median and Gaussian filters to remove salt-and-pepper noise from images effectively.  

3. **Image Restoration**:  
   - Restored damaged images iteratively by smoothing and inpainting using a mask.  

4. **Edge Detection**:  
   - Calculated horizontal and vertical derivatives using Sobel filters and visualized gradient magnitudes to highlight edges in images.  

5. **Canny Edge Detection**:  
   - Tuned parameters (low/high thresholds and sigma) to optimize edge detection and matched results with a target edge-detected image.  

These techniques demonstrate practical applications of spatial filtering and edge detection in image analysis.

# Assignment 3: Advanced Image Processing Techniques

This assignment explores a variety of advanced image processing tasks, including:

1. **Basic Bayer Pattern Reconstruction**:  
   - Reconstructed an RGB image from a GRGB Bayer pattern to simulate camera sensor data processing.

2. **Dynamic Palette Generation**:  
   - Utilized the KMeans clustering algorithm to calculate an optimal N-color palette for a given image and created a kd-tree palette for efficient color mapping.

3. **Geometric Transformations**:  
   - Implemented image rotation, scaling, skewing, and combined warping to manipulate and transform images effectively.

4. **Image Stitching**:  
   - Developed functionality to stitch two images together seamlessly, demonstrating the basics of panorama creation and alignment.

These tasks illustrate essential techniques for image reconstruction, optimization, and manipulation, foundational for computer vision and graphics applications.

# Assignment 4: Blob Detection and Watershed Segmentation

## **Part 1: Blob Detection**
1. **Gaussian Blur**:
   - Apply a Gaussian filter to reduce noise in the image.
   - Create a Difference of Gaussian (DoG) volume using multiple sigma values.

2. **Blob Center Detection**:
   - Identify regional minima in the DoG volume.
   - Collapse the 3D minima into a 2D binary image and overlay detected centers on the input image.

3. **Refinement**:
   - Use Li thresholding to filter out false positives.
   - Display refined blob centers.

## **Part 2: Watershed Segmentation**
1. **Regional Minima Detection**:
   - Label regional minima in the image.

2. **Iterative Label Propagation**:
   - Implement a minimum-following algorithm to propagate labels iteratively.

3. **Boundary Extraction**:
   - Apply watershed segmentation to detect cell boundaries.
   - Visualize the boundaries on the gradient-imposed image.

## **Usage**
Run the script to process the input `nuclei.png` and visualize blob centers and segmented regions.

# Assignment 5: JPEG Encoding and Decoding

The goal of this assignment was to implement the JPEG encoding and decoding process.

- **JPEG Encoding**: The image is first converted to the YCbCr color space, then divided into 8x8 blocks. Discrete Cosine Transform (DCT) is applied, followed by quantization. The quantized DCT coefficients are then reordered using a zig-zag scan. The compressed image and its size information are saved.

- **JPEG Decoding**: The compressed image and size information are loaded. The process involves inverse zig-zag scanning, inverse DCT, and dequantization to recover the image. Finally, the image is converted back to the RGB color space and displayed.

This process demonstrates the basic principles of image compression and evaluates the compression effect before and after.
