# Import libraries
from skimage import io, exposure, transform, color
from skimage.feature import ORB, match_descriptors
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform, SimilarityTransform, warp
import matplotlib.pyplot as plt
import numpy as np

def part4():
    ''' Stitch two images together '''
    image0 = io.imread('im1.jpg', True)
    image1 = io.imread('im2.jpg', True)

    plt.figure(figsize=(8, 12))
    plt.subplot(221), plt.imshow(image0, cmap='gray'), plt.title("First Image")
    plt.subplot(222), plt.imshow(image1, cmap='gray'), plt.title("Second Image")

    # -------- Feature detection and matching -----

    # Initiate ORB detector
    descriptor_extractor = ORB(n_keypoints=1000)

    # Find the keypoints and descriptors
    descriptor_extractor.detect_and_extract(image0)
    keypoints0 = descriptor_extractor.keypoints
    descriptors0 = descriptor_extractor.descriptors

    descriptor_extractor.detect_and_extract(image1)
    keypoints1 = descriptor_extractor.keypoints
    descriptors1 = descriptor_extractor.descriptors

    # Initialize Brute-Force matcher and exclude outliers. See match descriptor function.
    matches = match_descriptors(descriptors0, descriptors1, cross_check=True, max_distance=20)
    

    # -------- Transform estimation -------

    # Compute homography matrix using ransac and ProjectiveTransform
    # Extract matched keypoints
    data0 = keypoints0[matches[:, 0]][:, :2]
    data1 = keypoints1[matches[:, 1]][:, :2]

    # Update RANSAC call
    model_robust, inliers = ransac((data1, data0), ProjectiveTransform, min_samples=2, residual_threshold=2,max_trials = 2000)

    # ------------- Warping ----------------

    # Next, we produce the panorama itself. The first step is to find the shape of the output image
    # by considering the extents of all warped images.

    r, c = image1.shape[:2]

    # Note that transformations take coordinates in
    # (x, y) format, not (row, column), in order to be
    # consistent with most literature.
    corners = np.array([[0, 0],
                        [0, r],
                        [c, 0],
                        [c, r]])

    # Warp the image corners to their new positions.
    warped_corners = model_robust(corners)

    # Find the extents of both the reference image and the warped target image.
    all_corners = np.vstack((warped_corners, corners))

    corner_min = np.min(all_corners, axis=0)
    corner_max = np.max(all_corners, axis=0)

    # After calculating corner_min and corner_max
    output_shape = (corner_max - corner_min).astype(np.int32)
    output_shape = output_shape[::-1]  


    # ----- Note: The images are now warped according to the estimated transformation model.

    # A shift is added to ensure that both images are visible in their entirety.
    # Note that warp takes the inverse mapping as input.

    # For the first image, shift to bottom-left corner
    offset_first = SimilarityTransform(translation=[-corner_min[0], -corner_min[1]])

    # For the second image, add a small translation (adjust as needed)
    offset_second = SimilarityTransform(translation=[-corner_min[0] + 50, -corner_min[1] + 50])

    # Correct the offset application
    image0_warped = warp(image0, offset_first.inverse, output_shape=output_shape[::-1])
    image1_warped = warp(image1, (model_robust + offset_second).inverse, output_shape=output_shape[::-1])

    plt.subplot(223), plt.imshow(image0_warped, cmap="gray"), plt.title("Warped first image")
    plt.subplot(224), plt.imshow(image1_warped, cmap="gray"), plt.title("Warped second image")
    plt.show()

    # An alpha channel is added to the warped images before merging them into a single image:
    def add_alpha(image, background=-1):
        """Add an alpha layer to the image.
        The alpha layer is set to 1 for foreground
        and 0 for background.
        """
        rgb = color.gray2rgb(image)
        alpha = (image != background)
        return np.dstack((rgb, alpha))

    # Add alpha to the image0 and image1
    image0_alpha = add_alpha(image0)
    image1_alpha = add_alpha(image1)
    # Ensure the images have the same shape by padding or trimming
    max_rows = max(image0_alpha.shape[0], image1_alpha.shape[0])
    max_cols = max(image0_alpha.shape[1], image1_alpha.shape[1])

    # Function to adjust image size
    def adjust_image_size(image, target_rows, target_cols):
        # Pad the image if it's smaller than the target
        if image.shape[0] < target_rows or image.shape[1] < target_cols:
            padded_image = np.pad(image, ((0, target_rows - image.shape[0]), (0, target_cols - image.shape[1]), (0,0)), 'constant')
            return padded_image
        # Trim the image if it's larger than the target
        else:
            trimmed_image = image[:target_rows, :target_cols]
            return trimmed_image

    # Adjust sizes
    image0_alpha_adj = adjust_image_size(image0_alpha, max_rows, max_cols)
    image1_alpha_adj = adjust_image_size(image1_alpha, max_rows, max_cols)

    # merge them
    merged_image = (image0_alpha_adj[:, :, :3] * (1 - image0_alpha_adj[:, :, 3, np.newaxis]) +
                    image1_alpha_adj[:, :, :3] * (1 - image1_alpha_adj[:, :, 3, np.newaxis]) +
                    image0_alpha_adj[:, :, :3] * image0_alpha_adj[:, :, 3, np.newaxis] +
                    image1_alpha_adj[:, :, :3] * image1_alpha_adj[:, :, 3, np.newaxis]) / 2

    # The summed alpha layers give us an indication of
    # how many images were combined to make up each
    # pixel.  Divide by the number of images to get
    # an average.

    plt.figure(figsize=(10, 8))
    plt.imshow(merged_image, cmap="gray")
    plt.show()

    from skimage.feature import plot_matches
    # TODO: randomly select 10 inlier matches and show them using plot_matches
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))

    # Ensure that there are enough inliers to choose from
    if np.sum(inliers) >= 10:
        selected_matches = matches[inliers][:10]
        plot_matches(ax, image0, image1, keypoints0, keypoints1, selected_matches,only_matches = True)
        plt.show()

if __name__ == "__main__":
    part4()