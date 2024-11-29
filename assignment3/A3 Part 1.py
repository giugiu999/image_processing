# import statements
import numpy as np
from matplotlib import pyplot as plt
from skimage import io

def part1():
    """ BasicBayer: reconstruct RGB image using GRGB pattern"""
    filename_Grayimage = 'PeppersBayerGray.bmp'
    filename_gridB = 'gridB.bmp'  # for B
    filename_gridR = 'gridR.bmp'  # for R
    filename_gridG = 'gridG.bmp'  # for G

    # read image
    img = io.imread(filename_Grayimage, as_gray=True)
    h, w = img.shape

    # our final image will be a 3-dimensional image with 3 channels
    rgb = np.zeros((h, w, 3), np.uint8)

    # reconstruction of the green channel IG
    IG = np.copy(img)  # copy the image into each channel

    for row in range(0, h, 4):
        for col in range(0, w, 4):
            IG[row, col + 1] = (int(img[row, col]) + int(img[row, col + 2]) + int(img[row + 1, col + 1])) // 3  # B
            IG[row, col + 3] = (int(img[row, col+2]) + int(img[row + 1, col + 3])) // 2  # D
            IG[row+1, col] = (int(img[row, col]) + int(img[row+1, col + 1]) + int(img[row + 2, col])) // 3  # E
            IG[row + 1, col + 2] = (int(img[row, col + 2]) + int(img[row+1, col + 1]) + int(img[row + 1, col + 3]) +
                                    int(img[row + 2, col+2])) // 4  # G
            IG[row + 2, col + 1] = (int(img[row + 2, col]) + int(img[row + 3, col + 1]) + int(img[row+3, col+2]) + int(img[row + 1, col + 1])) // 4  # J
            IG[row+2, col + 3] = (int(img[row+2, col+2]) + int(img[row+1, col + 3]) + int(img[row + 3, col + 3])) // 3  # L
            IG[row +3, col] = (int(img[row+2,col]) + int(img[row+3,col+1])) //2 # M
            IG[row+3, col + 2] = (int(img[row+3, col+1]) + int(img[row+2, col + 2]) + int(img[row + 3, col + 3])) // 3  # O
            
    # show green (IG) in the first subplot (221) and add title
    plt.subplot(221)
    plt.imshow(IG, cmap='gray')
    plt.title('IG')

    # reconstruction of the red channel IR (similar to loops above)
    IR = np.copy(img)

    for row in range(0, h, 4):
        for col in range(0, w, 4):
            IR[row + 1, col + 1] = (int(img[row, col + 1]) + int(img[row + 2, col + 1])) // 2  # F
            IR[row, col + 2] = (int(img[row, col + 1]) + int(img[row, col + 3])) // 2  # C
            IR[row + 1, col + 2] = (int(img[row, col + 1]) + int(img[row, col + 3]) + int(img[row + 2, col + 1]) +
                                    int(img[row + 2, col + 3])) // 4  # G
            IR[row+1, col + 3] = (int(img[row, col+3]) + int(img[row+2, col+3])) //2 # H 
            IR[row+2, col +2 ] = (int(img[row+2, col+1]) + int(img[row + 2, col + 3])) //2 # K
            IR[row,col] = IR[row,col+1]   # copy the last second line value
            IR[row+1,col] = IR[row+1,col+1]
            IR[row+2,col] = IR[row+2,col+1]
            IR[row+3,col] = IR[row+3,col+1]
            IR[row+3,col] = IR[row+2,col] # copy the secondd col value  
            IR[row+3,col+1] = IR[row+2,col+1]
            IR[row+3,col+2] = IR[row+2,col+2]
            IR[row+3,col+3] = IR[row+2,col+3]
    # show IR in the second subplot (222) and title
    plt.subplot(222)
    plt.imshow(IR, cmap='gray')
    plt.title('IR')

    # reconstruction of the blue channel IB (similar to loops above)
    IB = np.copy(img)

    for row in range(0, h, 4):
        for col in range(0, w, 4):
            IB[row + 1, col + 1] = (int(img[row + 1, col]) + int(img[row + 1, col + 2])) // 2  # F
            IB[row + 2, col] = (int(img[row+1, col]) + int(img[row+3, col]) ) // 2  # I
            IB[row + 2, col + 1] = (int(img[row+1, col]) + int(img[row+1, col + 2]) + int(img[row + 3, col]) +
                                    int(img[row + 3, col+2])) // 4  # J
            IB[row+2,col+2] = (int(img[row+1,col+2]) + int(img[row + 3, col+2])) //2 # K
            IB[row +3, col + 1] = (int(img[row+3,col]) + int(img[row+3,col+2])) //2 # N
            IB[row,col] = IB[row+1,col] # copy the second row value
            IB[row,col+1] = IB[row+1,col+1]
            IB[row,col+2] = IB[row+1,col+2]
            IB[row,col+3] = IB[row+1,col+3]
            IB[row+1,col+3] = IB[row+1,col+2]
            IB[row+2,col+3] = IB[row+2,col+2]
            IB[row+3,col+3] = IB[row+3,col+2]

    # show IB in the third subplot (223) and title
    plt.subplot(223)
    plt.imshow(IB, cmap='gray')
    plt.title('IB')

    # merge the three channels IG, IB, IR in the correct order
    rgb[:, :, 1] = IG  # Green
    rgb[:, :, 0] = IR  # Red
    rgb[:, :, 2] = IB  # Blue

    # show rgb image in the final subplot (224) and add title
    plt.subplot(224)
    plt.imshow(rgb)
    plt.title('rgb')

    plt.show()

if __name__ == "__main__":
    part1()
