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

def part1_encoder():
    # Other imports and function definitions remain the same
    import matplotlib.pyplot as plt
    from skimage import io
    from scipy.fftpack import dct

    # Define the block size and read the image
    block_size = 8
    img = io.imread('bird.jpg')  # Ensure the path is correct for your setup

    # Convert the image from RGB space to YCbCr space
    def rgb_to_ycbcr(rgb_image):
        ycbcr_image = np.zeros_like(rgb_image, dtype=np.float32)
        ycbcr_image[:,:,0] = 0.299 * rgb_image[:,:,0] + 0.587 * rgb_image[:,:,1] + 0.114 * rgb_image[:,:,2]
        ycbcr_image[:,:,1] = 128 - 0.168736 * rgb_image[:,:,0] - 0.331264 * rgb_image[:,:,1] + 0.5 * rgb_image[:,:,2]
        ycbcr_image[:,:,2] = 128 + 0.5 * rgb_image[:,:,0] - 0.418688 * rgb_image[:,:,1] - 0.081312 * rgb_image[:,:,2]
        return ycbcr_image

    img_ycbcr = rgb_to_ycbcr(img)

    # Get the size of the image
    h, w, d = img_ycbcr.shape

    # Compute the number of 8x8 blocks
    nbh = int(np.ceil(h / block_size))
    nbw = int(np.ceil(w / block_size))

    # Pad the image
    H = nbh * block_size
    W = nbw * block_size
    padded_img = np.zeros((H, W, d))
    padded_img[0:h, 0:w, :] = img_ycbcr

    # Initialize the variables for counting non-zero DCT coefficients
    nonzeron_num = 0
    quantized_nonzeron_num = 0
    num_pixels = 0
    def dct2D(x):
        result = dct(dct(x.T, type = 2, norm='ortho').T, type = 2, norm='ortho')
        return result
    quantization_matrix_Y = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                                    [12, 12, 14, 19, 26, 58, 60, 55],
                                    [14, 13, 16, 24, 40, 57, 69, 56],
                                    [14, 17, 22, 29, 51, 87, 80, 62],
                                    [18, 22, 37, 56, 68, 109, 103, 77],
                                    [24, 35, 55, 64, 81, 104, 113, 92],
                                    [49, 64, 78, 87, 103, 121, 120, 101],
                                    [72, 92, 95, 98, 112, 100, 103, 99]])
    quantization_matrix_CbCr = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                                        [18, 21, 26, 66, 99, 99, 99, 99],
                                        [24, 26, 56, 99, 99, 99, 99, 99],
                                        [47, 66, 99, 99, 99, 99, 99, 99],
                                        [99, 99, 99, 99, 99, 99, 99, 99],
                                        [99, 99, 99, 99, 99, 99, 99, 99],
                                        [99, 99, 99, 99, 99, 99, 99, 99],
                                        [99, 99, 99, 99, 99, 99, 99, 99]])
    # Process each block
    for i in range(nbh):
        for j in range(nbw):
            for channel in range(d):  # Iterate over each color channel
                # Select the block from the padded image
                block = padded_img[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size, channel]

                # Apply 2D DCT
                
                dct_block = dct2D(block)

                # Quantization
                quant_matrix = quantization_matrix_Y if channel == 0 else quantization_matrix_CbCr
                quantized_dct_block = np.round(dct_block / quant_matrix)

                # Count non-zero elements in the quantized block
                nonzeron_num += np.count_nonzero(dct_block)
                quantized_nonzeron_num += np.count_nonzero(quantized_dct_block)

                # Reorder quantized DCT coefficients using zigzag function
                reordered_quantized_dct_block = zigzag(quantized_dct_block)

                # Reshape the reordered coefficients back into a block
                reshaped_quantized_block = np.reshape(reordered_quantized_dct_block, (block_size, block_size))

                # Place the quantized and reshaped block back into the padded image for visualization
                padded_img[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size, channel] = reshaped_quantized_block

    # Show the encoded (quantized) image - not the actual process but for visualization
    plt.imshow(np.uint8(padded_img), cmap='gray')
    plt.title('Encoded Image Visualization')
    plt.axis('off')
    plt.show()

    # Save the encoded image
    np.save('encoded.npy', padded_img)

    # Save the original image size and block size
    np.savetxt("size.txt", [h, w, d, block_size])

    # Calculate and print the compression ratio
    num_pixels = H * W * d
    before_compression = (nonzeron_num / num_pixels) * 100
    after_compression = (quantized_nonzeron_num / num_pixels) * 100
    print("Percentage of non-zero elements:")
    print("Before compression: {:.2f}%".format(before_compression))
    print("After compression: {:.2f}%".format(after_compression))

# Call the encoding function
part1_encoder()
