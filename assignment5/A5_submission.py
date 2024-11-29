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
	# JPEG encoding

    import numpy as np
    # import scipy
    import matplotlib.pyplot as plt
    from skimage import io
    from scipy.fftpack import dct,idct  


    # NOTE: Defining block size
    block_size = 8

    # TODO: Read image using skimage.io
    file = 'bird.jpg'
    img = io.imread(file)

    plt.imshow(img)
    plt.title('input image (RGB)')
    plt.axis('off')
    plt.show()
    
    # TODO: Convert the image from RGB space to YCbCr space
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


    '''
    Interesting property: Separability

    The separability property refers to the fact that a 2D DCT can be computed as the product of two 1D DCTs 
    applied along each dimension of the data independently. This means that a 2D DCT can be computed much more 
    efficiently as two 1D DCTs instead of directly computing the 2D transform.
    '''

    # TODO: Function to compute 2D Discrete Cosine Transform (DCT)
    # Apply DCT with type 2 and 'ortho' norm parameters

    def dct2D(x):
        dct_temp = dct(x, type=2, axis=0, norm='ortho')
        result = dct(dct_temp, type=2, axis=1, norm='ortho')
        return result


    # TODO: Get size of image
    h, w, d = img.shape
    # TODO: Compute number of blocks (of size 8-by-8), cast the numbers to int
    nbh = np.ceil(h / block_size).astype(int)
    nbw = np.ceil(w / block_size).astype(int)
    # TODO: (If necessary) Pad the image, get size of padded image
    H = nbh * block_size
    W = nbw * block_size
    # TODO: Create a numpy zero matrix with size of H,W,3 called padded img
    padded_img = np.zeros((H, W, d))

    # TODO: Copy the values of img into padded_img[0:h,0:w,:]
    padded_img[0:h, 0:w, :] = img

    # TODO: Display padded image
    plt.imshow(np.uint8(padded_img))
    plt.title('Padded Image')
    plt.axis('off')
    plt.show()


    # TODO: Create the quantization matrix
    # Refer to this https://www.sciencedirect.com/science/article/pii/S1742287608000285#:~:text=In%20the%20quantization%20stage%2C%20the,range%20from%200%20to%2065%2C535.&text=A%20lower%20number%20means%20that,higher%20quality%20image%20should%20result.
    # Use Fig. 1 as your quantization tables (1 table for Y channel and 1 table for Cb and Cr channels)
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

    # TODO: Initialize variables for compression calculations (only for the Y channel)
    quantized_blocks = np.zeros((H, W,3))
    nonzeron_num = 0
    quantized_nonzeron_num = 0
    num_pixels = 0

    # NOTE: Iterate over blocks
    for i in range(nbh):
        
        # Compute start and end row indices of the block
        row_ind_1 = i * block_size
        row_ind_2 = row_ind_1 + block_size
        
        for j in range(nbw):
            
            # Compute start and end column indices of the block
            col_ind_1 = j * block_size 
            col_ind_2 = col_ind_1 + block_size
            
            # TODO: Select current block to process using calculated indices (through slicing)
            Yblock = padded_img[row_ind_1:row_ind_2, col_ind_1:col_ind_2, 0]
            Cbblock = padded_img[row_ind_1:row_ind_2, col_ind_1:col_ind_2, 1]
            Crblock = padded_img[row_ind_1:row_ind_2, col_ind_1:col_ind_2, 2]
                
            # TODO: Apply dct2d() to selected block             
            YDCT = dct2D(Yblock)
            CbDCT = dct2D(Cbblock)
            CrDCT = dct2D(Crblock)

            # TODO: Quantization
            # Divide each element of DCT block by corresponding element in quantization matrix
            quantized_YDCT = np.round(YDCT / quantization_matrix_Y)
            quantized_CbDCT = np.round(CbDCT / quantization_matrix_CbCr)
            quantized_CrDCT = np.round(CrDCT / quantization_matrix_CbCr)

            # TODO: Reorder DCT coefficients into block (use zigzag function)
            reordered_Y = zigzag(quantized_YDCT)
            reordered_Cb = zigzag(quantized_CbDCT)
            reordered_Cr = zigzag(quantized_CrDCT)

            # TODO: Reshape reordered array to 8-by-8 2D block
            reshaped_Y = np.reshape(reordered_Y, (block_size, block_size))
            reshaped_Cb = np.reshape(reordered_Cb, (block_size, block_size))
            reshaped_Cr = np.reshape(reordered_Cr, (block_size, block_size))

            # TODO: Copy reshaped matrix into padded_img on current block corresponding indices
            padded_img[row_ind_1:row_ind_2, col_ind_1:col_ind_2, 0] = reshaped_Y
            padded_img[row_ind_1:row_ind_2, col_ind_1:col_ind_2, 1] = reshaped_Cb
            padded_img[row_ind_1:row_ind_2, col_ind_1:col_ind_2, 2] = reshaped_Cr

            # TODO: Compute pixel locations with non-zero values before and after quantization (only in Y channel)
            if i == 0 and j == 0:  # Do this for the first block only, to initialize the count
                nonzeron_num = np.count_nonzero(YDCT)
                quantized_nonzeron_num = np.count_nonzero(quantized_YDCT)
            else:
                nonzeron_num += np.count_nonzero(YDCT)
                quantized_nonzeron_num += np.count_nonzero(quantized_YDCT)
            # TODO: Compute total number of pixels
            num_pixels = nbh * nbw * block_size * block_size
    plt.imshow(np.uint8(padded_img))
    plt.title('encoded image')
    plt.axis('off')
    plt.show()
                

    # NOTE: Write h, w, c, block_size and padded_img into .txt files at the end of encoding

    # TODO: Write padded_img into 'encoded.npy' file (use np.save() function)
    # First parameter should be 'encoded.npy'
    np.save('encoded.npy', padded_img)

    # TODO: write [h, w, c, block_size] into size.txt
    # First parameter should be 'size.txt' (use np.savetxt() function)
    np.savetxt('size.txt', [h, w, d, block_size])

    # TODO: Calculate percentage of pixel locations with non-zero values before and after to measure degree of compression 
    before = (nonzeron_num / num_pixels) * 100
    after = (quantized_nonzeron_num / num_pixels) * 100
    # Print statements as shown in eClass
    print("Percentage of non-zero elements in Luma channel:")
    print("Before compression: {}%".format(before))
    print("After compression: {}%".format(after))

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
        temp = idct(x, type=2, norm='ortho').T
        result = idct(temp, type=2, norm='ortho').T
        return result


    # TODO: Load 'encoded.npy' into padded_img (using np.load() function)
    padded_img = np.load('encoded.npy')

    # TODO: Load h, w, c, block_size and padded_img from the size.txt file
    [h, w, c, block_size] = np.loadtxt('size.txt')

    # TODO: 6. Get size of padded_img, cast to int if needed
    [H,W,d] = padded_img.shape
    h = int(h)
    w = int(w)
    block_size = int(block_size)

    # TODO: Create the quantization matrix (Same as before)
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

    # TODO: Compute number of blocks (of size 8-by-8), cast to int
    nbh = int(np.ceil(H/block_size)) # (number of blocks in height)
    nbw = int(np.ceil(W/block_size)) # (number of blocks in width)

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
                Yblock = padded_img[row_ind_1:row_ind_2, col_ind_1:col_ind_2, 0]  # Y component
                Cbblock = padded_img[row_ind_1:row_ind_2, col_ind_1:col_ind_2, 1]  # Cb component
                Crblock = padded_img[row_ind_1:row_ind_2, col_ind_1:col_ind_2, 2]
                
                # TODO: Reshape 8-by-8 2D block to 1D array
                Yreshaped = Yblock.flatten()
                Cbreshaped = Cbblock.flatten()
                Crreshaped = Crblock.flatten()
                
                # TODO: Reorder array into block (use inverse_zigzag function)
                Yreordered = inverse_zigzag(Yreshaped, block_size, block_size)
                Cbreordered =inverse_zigzag(Cbreshaped, block_size, block_size)
                Crreordered =inverse_zigzag(Crreshaped, block_size, block_size)
                
                # TODO: De-quantization
                # Multiply each element of reordered block by corresponding element in quantization matrix
                dequantized_YDCT = np.multiply(Yreordered, quantization_matrix_Y)
                dequantized_CbDCT =np.multiply(Cbreordered, quantization_matrix_CbCr)
                dequantized_CrDCT =np.multiply(Crreordered, quantization_matrix_CbCr)
                
                # TODO: Apply idct2d() to reordered matrix 
                YIDCT = idct2D(dequantized_YDCT)
                CbIDCT = idct2D(dequantized_CbDCT)
                CrIDCT = idct2D(dequantized_CrDCT)

                # TODO: Copy IDCT matrix into padded_img on current block corresponding indices
                padded_img[ row_ind_1:row_ind_2 ,col_ind_1:col_ind_2,0 ] = YIDCT
                padded_img[ row_ind_1:row_ind_2 ,col_ind_1:col_ind_2,1 ] = CbIDCT
                padded_img[ row_ind_1:row_ind_2 ,col_ind_1:col_ind_2,2 ] = CrIDCT
    # TODO: Remove out-of-range values
    padded_img = np.clip(padded_img, 0, 255).astype(np.uint8)


    plt.imshow(np.uint8(padded_img))
    plt.title('decoded padded image (YCbCr)')
    plt.axis('off')
    plt.show()

    # TODO: Get original sized image from padded_img
    decoded_img = padded_img[0:h, 0:w,:]

    plt.imshow(np.uint8(decoded_img))
    plt.title('decoded padded image (YCbCr)')
    plt.axis('off')
    plt.show()
    
    # TODO: Convert the image from YCbCr to RGB
    def convert_to_rgb(ycbcr_image):
        xform = np.array([[1.0, 0.0, 1.40210],
                        [1.0, -0.34414, -0.71414],
                        [1.0, 1.77180, 0.0]])
        
        rgb_image = ycbcr_image.astype(np.float64)
        rgb_image[..., [1, 2]] -= 128

        rgb_image = np.dot(rgb_image, xform.T)

        rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)

        return rgb_image
    
    decoded_img_rgb = convert_to_rgb(decoded_img)
    # TODO: Remove out-of-range values
    decoded_img = np.clip(decoded_img_rgb, 0, 255)
    
    plt.imshow(np.uint8(decoded_img))
    plt.title('decoded image (RGB)')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    part1_encoder()
    part2_decoder()

