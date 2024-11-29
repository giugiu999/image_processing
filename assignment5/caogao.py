'''
###### DO NOT EDIT ######
(Scroll down for start of the assignment)

# MATLAB Code:
# Alexey S. Sokolov a.k.a. nICKEL, Moscow, Russia
# June 2007
# alex.nickel@gmail.com

Zig-zag section
'''

import numpy as np

# Zigzag scan of a matrix

# --INPUT--
# Argument: 2D matrix of any size, not strictly square 

# --OUTPUT--
# Returns: 1-by-(m*n) array, where input matrix is m*n

def zigzag(input):
    #initializing the variables
    #----------------------------------
    h = 0
    v = 0

    vmin = 0
    hmin = 0

    vmax = input.shape[0]
    hmax = input.shape[1]
    
    #print(vmax ,hmax )

    i = 0

    output = np.zeros(( vmax * hmax))
    #----------------------------------

    while ((v < vmax) and (h < hmax)):
        
        if ((h + v) % 2) == 0:                 # going up
            
            if (v == vmin):
                #print(1)
                output[i] = input[v, h]        # if we got to the first line

                if (h == hmax):
                    v = v + 1
                else:
                    h = h + 1                        

                i = i + 1

            elif ((h == hmax -1 ) and (v < vmax)):   # if we got to the last column
                #print(2)
                output[i] = input[v, h] 
                v = v + 1
                i = i + 1

            elif ((v > vmin) and (h < hmax -1 )):    # all other cases
                #print(3)
                output[i] = input[v, h] 
                v = v - 1
                h = h + 1
                i = i + 1

        
        else:                                    # going down

            if ((v == vmax -1) and (h <= hmax -1)):       # if we got to the last line
                #print(4)
                output[i] = input[v, h] 
                h = h + 1
                i = i + 1
        
            elif (h == hmin):                  # if we got to the first column
                #print(5)
                output[i] = input[v, h] 

                if (v == vmax -1):
                    h = h + 1
                else:
                    v = v + 1

                i = i + 1

            elif ((v < vmax -1) and (h > hmin)):     # all other cases
                #print(6)
                output[i] = input[v, h] 
                v = v + 1
                h = h - 1
                i = i + 1


        if ((v == vmax-1) and (h == hmax-1)):          # bottom right element
            #print(7)        	
            output[i] = input[v, h] 
            break

    #print ('v:',v,', h:',h,', i:',i)
    return output


# Inverse zigzag scan of a matrix

# --INPUT--
# Argument: 1-by-m*n array, m & n are vertical & horizontal sizes of output matrix

# --OUTPUT--
# Returns: a 2D matrix of defined sizes with input array items gathered by zigzag

def inverse_zigzag(input, vmax, hmax):

    #print input.shape

    # initializing the variables
    #----------------------------------
    h = 0
    v = 0

    vmin = 0
    hmin = 0

    output = np.zeros((vmax, hmax))

    i = 0
    #----------------------------------

    while ((v < vmax) and (h < hmax)): 
        #print ('v:',v,', h:',h,', i:',i)   	
        if ((h + v) % 2) == 0:                 # going up
            
            if (v == vmin):
                #print(1)
                
                output[v, h] = input[i]        # if we got to the first line

                if (h == hmax):
                    v = v + 1
                else:
                    h = h + 1                        

                i = i + 1

            elif ((h == hmax -1 ) and (v < vmax)):   # if we got to the last column
                #print(2)
                output[v, h] = input[i] 
                v = v + 1
                i = i + 1

            elif ((v > vmin) and (h < hmax -1 )):    # all other cases
                #print(3)
                output[v, h] = input[i] 
                v = v - 1
                h = h + 1
                i = i + 1

        
        else:                                    # going down

            if ((v == vmax -1) and (h <= hmax -1)):       # if we got to the last line
                #print(4)
                output[v, h] = input[i] 
                h = h + 1
                i = i + 1
        
            elif (h == hmin):                  # if we got to the first column
                #print(5)
                output[v, h] = input[i] 
                if (v == vmax -1):
                    h = h + 1
                else:
                    v = v + 1
                i = i + 1
                                
            elif((v < vmax -1) and (h > hmin)):     # all other cases
                output[v, h] = input[i] 
                v = v + 1
                h = h - 1
                i = i + 1


        if ((v == vmax-1) and (h == hmax-1)):          # bottom right element
            #print(7)        	
            output[v, h] = input[i] 
            break


    return output


'''
######
Assignment 5 starts here
######
'''

def part1_encoder():
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage import io
    from scipy.fftpack import dct

    # Defining block size
    block_size = 8

    # Read image using skimage.io
    file = 'bird.jpg'
    img = io.imread(file)

    plt.imshow(img)
    plt.title('input image (RGB)')
    plt.axis('off')
    plt.show()

    # Convert the image from RGB space to YCbCr space
    def rgb_to_ycbcr(rgb_image):
        ycbcr_image = np.zeros_like(rgb_image, dtype=np.float32)
        ycbcr_image[:, :, 0] = 0.299 * rgb_image[:, :, 0] + 0.587 * rgb_image[:, :, 1] + 0.114 * rgb_image[:, :, 2]
        ycbcr_image[:, :, 1] = 128 - 0.168736 * rgb_image[:, :, 0] - 0.331264 * rgb_image[:, :, 1] + 0.5 * rgb_image[:, :, 2]
        ycbcr_image[:, :, 2] = 128 + 0.5 * rgb_image[:, :, 0] - 0.418688 * rgb_image[:, :, 1] - 0.081312 * rgb_image[:, :, 2]
        return ycbcr_image

    img = rgb_to_ycbcr(img)

    plt.imshow(np.uint8(img))
    plt.title('input image (YCbCr)')
    plt.axis('off')
    plt.show()

    # Function to compute 2D Discrete Cosine Transform (DCT)
    def dct2D(x):
        return dct(dct(x.T, norm='ortho').T, norm='ortho')

    # Get size of image
    h, w, d = img.shape

    # Compute number of blocks (of size 8-by-8), cast the numbers to int
    nbh = np.ceil(h / block_size).astype(int)
    nbw = np.ceil(w / block_size).astype(int)

    # Pad the image
    H = nbh * block_size
    W = nbw * block_size
    padded_img = np.zeros((H, W, d))
    padded_img[0:h, 0:w, :] = img

    plt.imshow(np.uint8(padded_img))
    plt.title('Padded Image')
    plt.axis('off')
    plt.show()

    # Create the quantization matrix
    # Create the quantization matrix for Y channel
    quantization_matrix_Y = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])

    # Create the quantization matrix for Cb and Cr channels
    quantization_matrix_CbCr = np.array([
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 13, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99]
    ])

    # Initialize variables for compression calculations
    quantized_blocks = np.zeros((H, W,3))
    nonzeron_num = 0
    quantized_nonzeron_num = 0
    num_pixels = 0

    for i in range(nbh):
        for j in range(nbw):
            # Process each block
            for channel in range(3):
                # Apply DCT
                block = padded_img[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size, channel]
                dct_block = dct2D(block)
                
                # Quantize
                quant_matrix = quantization_matrix_Y if channel == 0 else quantization_matrix_CbCr
                quantized_block = np.round(dct_block / quant_matrix)
                
                # Calculate non-zero elements
                if channel == 0:  # Only for Y channel
                    nonzeron_num += np.count_nonzero(dct_block)
                    quantized_nonzeron_num += np.count_nonzero(quantized_block)

                # Save quantized blocks (for demonstration, we're not using it for encoding)
                # In actual JPEG encoding, you would proceed with further compression steps here

    # Calculate compression metrics
    num_pixels = H * W  # Correcting to consider the entire image size
    before = nonzeron_num / num_pixels
    after = quantized_nonzeron_num / num_pixels
        # Adjusted Code for Visualization without Inverse DCT
    for i in range(nbh):
        for j in range(nbw):
            for k in range(3):  # Iterate over color channels
                block = padded_img[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size, k]
                
                # Apply DCT
                dct_block = dct2D(block - 128)  # Centering around 0 by subtracting 128
                
                # Quantize
                if k == 0:  # Y channel
                    quantized_block = np.round(dct_block / quantization_matrix_Y)
                else:  # Cb or Cr channel
                    quantized_block = np.round(dct_block / quantization_matrix_CbCr)
                
                # Calculate non-zero elements before and after quantization for Y channel only
                if k == 0:
                    nonzeron_num += np.count_nonzero(dct_block)
                    quantized_nonzeron_num += np.count_nonzero(quantized_block)

                # Perform inverse DCT for visualization (skipping to match your request)
                # In a real scenario, you would inverse DCT here to reconstruct the image block
                
                # Store quantized (not correctly visualized without IDCT) blocks back for visualization
                # This is incorrect for a real JPEG process but follows your structure
                quantized_blocks[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size, k] = quantized_block

    # Visualize "encoded" image (incorrect without IDCT)
    plt.imshow(np.uint8(quantized_blocks[:,:,0]), cmap='gray')  # Displaying Y channel only for clarity
    plt.title('Encoded Image Visualization')
    plt.axis('off')
    plt.show()

    # NOTE: This visualization is not accurate for JPEG encoding as it skips inverse DCT.
    # It's meant to demonstrate quantization's effect on blocks.

    # Print compression results
    print("Percentage of non-zero elements:")
    print(f"Before compression: {before*100:.2f}%")
    print(f"After compression: {after*100:.2f}%")



'''
def part2_decoder():
    # JPEG decoding

    import numpy as np
    # import scipy
    import matplotlib.pyplot as plt
    from skimage import io
    from scipy.fftpack import dct,idct 

    # NOTE: Defining block size
    block_size = 8 

    # TODO: Function to compute 2D Discrete Cosine Transform (DCT)
    # Apply IDCT with type 2 and 'ortho' norm parameters

    def idct2D(x):
        ###### Your code here ######
        return result


    # TODO: Load 'encoded.npy' into padded_img (using np.load() function)
    ###### Your code here ######

    # TODO: Load h, w, c, block_size and padded_img from the size.txt file
    ###### Your code here ######

    # TODO: 6. Get size of padded_img, cast to int if needed
    ###### Your code here ######

    # TODO: Create the quantization matrix (Same as before)
    quantization_matrix_Y = # quantization table for the Y channel
    quantization_matrix_CbCr = # quantization table for the Y channel
    
    ###### Your code here ######

    # TODO: Compute number of blocks (of size 8-by-8), cast to int
    nbh = ###### Your code here ###### # (number of blocks in height)
    nbw = ###### Your code here ###### # (number of blocks in width)

    # TODO: iterate over blocks
    for i in range(nbh):
        
            # Compute start and end row indices of the block
            row_ind_1 = i * block_size
            
            row_ind_2 = row_ind_1 + block_size
            
            for j in range(nbw):
                
                # Compute start and end column indices of the block
                col_ind_1 = j * block_size

                col_ind_2 = col_ind_1 + block_size
                
                # TODO: Select current block to process using calculated indices
                Yblock = ###### Your code here ######
                Cbblock = ###### Your code here ######
                Crblock = ###### Your code here ######
                
                # TODO: Reshape 8-by-8 2D block to 1D array
                Yreshaped = ###### Your code here ######
                Cbreshaped = ###### Your code here ######
                Crreshaped = ###### Your code here ######
                
                # TODO: Reorder array into block (use inverse_zigzag function)
                Yreordered = ###### Your code here ######
                Cbreordered = ###### Your code here ######
                Crreordered = ###### Your code here ######
                
                # TODO: De-quantization
                # Multiply each element of reordered block by corresponding element in quantization matrix
                dequantized_YDCT = ###### Your code here ######
                dequantized_CbDCT = ###### Your code here ######
                dequantized_CrDCT = ###### Your code here ######
                
                # TODO: Apply idct2d() to reordered matrix 
                YIDCT = ###### Your code here ######
                CbIDCT = ###### Your code here ######
                CrIDCT = ###### Your code here ######

                # TODO: Copy IDCT matrix into padded_img on current block corresponding indices
                ###### Your code here ######

    # TODO: Remove out-of-range values
    ###### Your code here ######

    plt.imshow(np.uint8(padded_img))
    plt.title('decoded padded image (YCbCr)')
    plt.axis('off')
    plt.show()

    # TODO: Get original sized image from padded_img
    ###### Your code here ######

    plt.imshow(np.uint8(decoded_img))
    plt.title('decoded padded image (YCbCr)')
    plt.axis('off')
    plt.show()
    
    # TODO: Convert the image from YCbCr to RGB
    ###### Your code here ######
    
    # TODO: Remove out-of-range values
    ###### Your code here ######
    
    plt.imshow(np.uint8(decoded_img))
    plt.title('decoded image (RGB)')
    plt.axis('off')
    plt.show()

'''
if __name__ == '__main__':
    part1_encoder()
    # part2_decoder()

