# Import libraries
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import warp
from skimage import img_as_float

def read_image():
    original_img = io.imread('bird.jpeg')
    return original_img

def calculate_trans_mat(image):
    """
    return translation matrix that shifts center of image to the origin and its inverse
    """
    trans_mat = None
    trans_mat_inv = None

    # TODO: implement this function (overwrite the two lines above)
    trans_mat = np.array([[1, 0, image.shape[1] / 2],
                        [0, 1, image.shape[0] / 2],
                        [0, 0, 1]])

    trans_mat_inv = np.linalg.inv(trans_mat)
    
    return trans_mat, trans_mat_inv

def rotate_image(image):
    ''' rotate and return image '''
    h, w = image.shape[:2]
    trans_mat, trans_mat_inv = calculate_trans_mat(image)

    # TODO: determine angle and create Tr
    angle = -75
    angle_rad = np.radians(angle)
    Tr = np.array([[np.cos(angle_rad), -np.sin(angle_rad), 0.0],
                   [np.sin(angle_rad), np.cos(angle_rad), 0.0],
                   [0.0, 0.0, 1.]])

    # TODO: compute inverse transformation to go from output to input pixel locations
    Tr_inv = trans_mat.dot(np.linalg.inv(Tr)).dot(trans_mat_inv)

    out_img = np.zeros_like(image)
    for out_y in range(h):
        for out_x in range(w):
            # TODO: find input pixel location from output pixel location and inverse transform matrix, copy over value from input location to output location
            in_loc = Tr_inv.dot([out_x, out_y, 1])[:2]
            in_x, in_y = map(int, in_loc)
            # Check if the input location is within bounds 
            if 0 <= in_x < w and 0 <= in_y < h:
                # Copy over value from input location to output location
                out_img[out_y, out_x, :] = image[in_y, in_x, :]  

    return out_img, Tr.astype(float)

def scale_image(image):
    ''' scale image and return '''
    # TODO: implement this function, similar to above
    h, w = image.shape[:2]
    tras_mat,tras_mat_inv = calculate_trans_mat(image)

    # Determine scale factors
    scale_x = 1.5
    scale_y = 2.5

    Ts = np.array([[scale_x, 0, 0],
                   [0, scale_y, 0],
                   [0, 0, 1]])

    Ts_inv = tras_mat.dot(np.linalg.inv(Ts)).dot(tras_mat_inv)

    out_img = np.zeros_like(image)
    for out_y in range(h):
        for out_x in range(w):
            in_loc = Ts_inv.dot([out_x, out_y, 1])[:2]
            in_x, in_y = map(int, in_loc)
            if 0 <= in_x < w and 0 <= in_y < h:
                out_img[out_y, out_x, :] = image[in_y, in_x, :]

    return out_img, Ts.astype(float)

def skew_image(image):
    ''' Skew image and return '''
    # TODO: implement this function like above
    h, w = image.shape[:2]
    tras_mat,tras_mat_inv = calculate_trans_mat(image)
    # Skew factors
    skew_x = 0.2
    skew_y = 0.2
    
    Tskew = np.array([[1, skew_x, 0],
                      [skew_y, 1, 0],
                      [0, 0, 1]])

    Tskew_inv = tras_mat.dot(np.linalg.inv(Tskew)).dot(tras_mat_inv)

    out_img = np.zeros_like(image)
    for out_y in range(h):
        for out_x in range(w):
            in_loc = Tskew_inv.dot([out_x, out_y, 1])[:2]
            in_x, in_y = map(int, in_loc)
            if 0 <= in_x < w and 0 <= in_y < h:
                out_img[out_y, out_x, :] = image[in_y, in_x, :]

    return out_img, Tskew.astype(float)


def combined_warp(image):
    ''' implement your own code to perform the combined warp of rotate, scale, skew and return image + transformation matrix  '''
    # TODO: implement combined warp on your own. 
    # You need to combine the transformation matrices before performing the warp
    # (you may want to use the above functions to get the transformation matrices)
    _, Tr = rotate_image(image)
    _, Ts = scale_image(image)
    _, Tskew = skew_image(image)

    # Combine transformations
    Tc = Ts.dot(Tr).dot(Tskew)

    tras_mat,tras_mat_inv = calculate_trans_mat(image)
    Tcombined_inv = tras_mat.dot(np.linalg.inv(Tc)).dot(tras_mat_inv)

    # Apply combined transformation
    h, w = image.shape[:2]
    out_img = np.zeros_like(image)
    for out_y in range(h):
        for out_x in range(w):
            in_loc = Tcombined_inv.dot([out_x, out_y, 1])[:2]
            in_x, in_y = map(int, in_loc)
            if 0 <= in_x < w and 0 <= in_y < h:
                out_img[out_y, out_x, :] = image[in_y, in_x, :]

    return out_img, Tc
    

def combined_warp_biinear(image):
    _, Tr = rotate_image(image)
    _, Ts = scale_image(image)
    _, Tskew = skew_image(image)
    tras_mat, tras_mat_inv = calculate_trans_mat(image)

    # Combine transformations
    Tc = Ts.dot(Tr).dot(Tskew)

    # Apply combined transformation with bilinear interpolation
    tras_inv = tras_mat.dot(np.linalg.inv(Tc)).dot(tras_mat_inv)

    # Apply the combined transformation with bilinear interpolation
    out_img = warp(image, inverse_map=tras_inv, order=1, cval=0.0, preserve_range=True)

    return out_img


if __name__ == "__main__":
    image = read_image()
    plt.imshow(image), plt.title("Oiginal Image"), plt.show()

    rotated_img, _ = rotate_image(image)
    plt.figure(figsize=(15,5))
    plt.subplot(131),plt.imshow(rotated_img), plt.title("Rotated Image")

    scaled_img, _ = scale_image(image)
    plt.subplot(132),plt.imshow(scaled_img), plt.title("Scaled Image")

    skewed_img, _ = skew_image(image)
    plt.subplot(133),plt.imshow(skewed_img), plt.title("Skewed Image"), plt.show()

    combined_warp_img, _ = combined_warp(image)
    plt.figure(figsize=(10,5))
    plt.subplot(121),plt.imshow(combined_warp_img), plt.title("Combined Warp Image")

    combined_warp_biliear_img = combined_warp_biinear(image)
    plt.subplot(122),plt.imshow(combined_warp_biliear_img.astype(np.uint8)), plt.title("Combined Warp Image with Bilinear Interpolation"),plt.show()



