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

    # TODO: Read image using skimage.io as grayscale
    ###### Your code here ######
    img = io.imread("cameraman.png", as_gray=True)

    plt.imshow(img,cmap='gray')
    plt.title('input image')
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
        ###### Your code here ######
        #作业中提供的链接https://cs.stanford.edu/people/eroberts/courses/soco/projects/data-compression/lossy/jpeg/coeff.htm
        #有该算法的相关介绍，关于实现方式可以参考https://stackoverflow.com/questions/15978468/using-the-scipy-dct-function-to-create-a-2d-dct-ii
        temp = dct(x, type=2, norm='ortho').T
        result = dct(temp, type=2, norm='ortho').T
        return result


    # TODO: Get size of image
    ###### Your code here ######
    #todo的介绍十分容易理解，不再重复阐述同样的意思
    [h ,w] = img.shape

    # TODO: Compute number of blocks (of size 8-by-8), cast the numbers to int
    #np.ceil用于向上取整
    nbh = int(np.ceil(h / block_size))###### Your code here ###### # (number of blocks in height)
    nbw = int(np.ceil(w / block_size)) ###### Your code here ###### # (number of blocks in width)


    # TODO: (If necessary) Pad the image, get size of padded image
    H = block_size * nbh###### Your code here ######  # height of padded image
    W = block_size * nbw###### Your code here ######  # width of padded image

    # TODO: Create a numpy zero matrix with size of H,W called padded img
    padded_img = np.zeros((H,W))###### Your code here ######

    # TODO: Copy the values of img into padded_img[0:h,0:w]
    ###### Your code here ######
    padded_img[0:h,0:w] = np.copy(img)

    # TODO: Display padded image
    plt.imshow(np.uint8(padded_img),cmap='gray')
    plt.title("input padded image")
    plt.axis('off')
    plt.show()


    # TODO: Create the quantization matrix
    # Refer to this PDF (https://www.ijg.org/files/Wallace.JPEG.pdf) 
    # Use Fig. 10 (c) (Page 12) as your quantization matrix
    
    ###### Your code here ######
    Q = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                  [12, 12, 14, 19, 26, 58, 60, 55],
                  [14, 13, 16, 24, 40, 57, 69, 56],
                  [14, 17, 22, 29, 51, 87, 80, 62],
                  [18, 22, 37, 56, 68, 109, 103, 77],
                  [24, 35, 55, 64, 81, 104, 113, 92],
                  [49, 64, 78, 87, 103, 121, 120, 101],
                  [72, 92, 95, 98, 112, 100, 103, 99]])

    # TODO: Initialize an empty numpy array to store the quantized blocks
    quantized_blocks = np.zeros((H, W))###### Your code here ######

    '''NEW ADDITIONS/MODIFICATIONS'''
    '''NEW ADDITIONS/MODIFICATIONS'''
    '''NEW ADDITIONS/MODIFICATIONS'''

    # TODO: Initialize variables for compression calculations
    ###### Your code here ######
    nzn = 0 #non-zero number
    quantized_nzn = 0 #quantized non-zero number
    nop = 0 #number of pixels

    # NOTE: Iterate over blocks
    for i in range(nbh):
        
        # Compute start and end row indices of the block
        row_ind_1 = i * block_size
        row_ind_2 = row_ind_1 + block_size
        
        for j in range(nbw):
            
            # Compute start and end column indices of the block
            col_ind_1 = j * block_size 
            col_ind_2 = col_ind_1 + block_size
            
            # TODO: Select current block to process using calculated indices (through splicing)
            block = padded_img[row_ind_1:row_ind_2 ,col_ind_1:col_ind_2 ]###### Your code here ######
            
            # TODO: Apply dct2d() to selected block
            DCT = dct2D(block)###### Your code here ######

            # TODO: Quantization
            # Divide each element of DCT block by corresponding element in quantization matrix
            #参见作业中提供的链接https://cs.stanford.edu/people/eroberts/courses/soco/projects/data-compression/lossy/jpeg/coeff.htm
            quantized_DCT = np.round(np.divide(DCT,Q))###### Your code here ######

            # TODO: Reorder DCT coefficients into block (use zigzag function)
            #由于后续计算压缩比的两个数值，经测试分别是有量化操作的图片编码与无量化操作的图片编码，因此此处计算了两个数值
            reordered = zigzag(DCT)###### Your code here ######
            quantized_reordered = zigzag(quantized_DCT)

            # TODO: Reshape reordered array to 8-by-8 2D block
            #同上，分别为正常编码与量化编码
            reshaped = np.reshape(reordered, (block_size, block_size))###### Your code here ######
            quantized_reshaped = np.reshape(quantized_reordered, (block_size, block_size))

            # TODO: Copy reshaped matrix into padded_img on current block corresponding indices
            ###### Your code here ######
            #将正常编码与量化编码存储起来，正常编码的结果为padded_img，量化编码的结果为quantized_blocks
            padded_img[row_ind_1:row_ind_2 , col_ind_1:col_ind_2] = reshaped
            quantized_blocks[row_ind_1:row_ind_2 , col_ind_1:col_ind_2] = quantized_reshaped

            '''NEW ADDITIONS/MODIFICATIONS'''
            '''NEW ADDITIONS/MODIFICATIONS'''
            '''NEW ADDITIONS/MODIFICATIONS'''

            # TODO: Compute pixel locations with non-zero values before and after quantization
            # TODO: Compute total number of pixels
            ###### Your code here ####
            #np.count_nonzero用于计算图像中非0值的像素个数，并将其除以像素数即为压缩比
            nzn += np.count_nonzero(reshaped)
            quantized_nzn += np.count_nonzero(quantized_reshaped)
            nop += block_size*block_size

    #原输出为padded_img，但是按照作业中显示的应该是量化操作的结果，因此修改为quantized_blocks
    plt.imshow(np.uint8(quantized_blocks),cmap='gray')
    plt.title("encoded image")
    plt.axis('off')
    plt.show()
                

    # NOTE: Write h, w, block_size and padded_img into .txt files at the end of encoding

    # TODO: Write padded_img into 'encoded.txt' file
    # First parameter should be 'encoded.txt'
    ###### Your code here ######
    #注释中希望保存的应该是量化的图像，因此此处保存quantized_blocks
    np.savetxt("encoded.txt", quantized_blocks)

    # TODO: write [h, w, block_size] into size.txt
    # First parameter should be 'size.txt'
    ###### Your code here ######
    np.savetxt("size.txt", [h, w, block_size])

    '''NEW ADDITIONS/MODIFICATIONS'''
    '''NEW ADDITIONS/MODIFICATIONS'''
    '''NEW ADDITIONS/MODIFICATIONS'''

    # TODO: Calculate percentage of pixel locations with non-zero values before and after to measure degree of compression 

    before = nzn/nop###### Your code here ######
    after = quantized_nzn/nop###### Your code here ######

    # Print statements as shown in eClass
    ###### Your code here ######
    print("Percentage of non-zero elements:")
    print("Before compression:{}%".format(before*100))
    print("After compression:{}%".format(after*100))




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
        #同dct2D
        temp = idct(x, type=2, norm='ortho').T
        result = idct(temp, type=2, norm='ortho').T
        return result


    # TODO: Load 'encoded.txt' into padded_img
    ###### Your code here ######
    padded_img = np.loadtxt('encoded.txt')

    # TODO: Load h, w, block_size and padded_img from .txt files
    ###### Your code here ######
    [h, w, block_size] = np.loadtxt('size.txt')

    # TODO: 6. Get size of padded_img, cast to int if needed
    ###### Your code here ######
    [H,W] = padded_img.shape
    #因为加载的h, w, block_size都为浮点数，影响后续操作执行，因此将其转为整型
    h = int(h)
    w = int(w)
    block_size = int(block_size)

    # TODO: Create the quantization matrix (Same as before)
    # Refer to this PDF (https://www.ijg.org/files/Wallace.JPEG.pdf 
    # Use Fig. 10 (c) (Page 12) as your quantization matrix
    
    ###### Your code here ######
    Q = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                  [12, 12, 14, 19, 26, 58, 60, 55],
                  [14, 13, 16, 24, 40, 57, 69, 56],
                  [14, 17, 22, 29, 51, 87, 80, 62],
                  [18, 22, 37, 56, 68, 109, 103, 77],
                  [24, 35, 55, 64, 81, 104, 113, 92],
                  [49, 64, 78, 87, 103, 121, 120, 101],
                  [72, 92, 95, 98, 112, 100, 103, 99]])

    # TODO: Compute number of blocks (of size 8-by-8), cast to int
    nbh = int(np.ceil(H/block_size))###### Your code here ###### # (number of blocks in height)
    nbw = int(np.ceil(W/block_size))###### Your code here ###### # (number of blocks in width)

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
                block = padded_img[row_ind_1:row_ind_2 ,col_ind_1:col_ind_2 ]###### Your code here ######
                
                # TODO: Reshape 8-by-8 2D block to 1D array
                #将2维展开成1维
                reshaped = block.flatten()###### Your code here ######
                
                # TODO: Reorder array into block (use inverse_zigzag function)
                reordered = inverse_zigzag(reshaped, block_size, block_size)###### Your code here ######
                
                # TODO: De-quantization
                # Multiply each element of reordered block by corresponding element in quantization matrix
                dequantized_DCT = np.multiply(reordered, Q)###### Your code here ######
                
                # TODO: Apply idct2d() to reordered matrix 
                IDCT = idct2D(dequantized_DCT)###### Your code here ######

                # TODO: Copy IDCT matrix into padded_img on current block corresponding indices
                ###### Your code here ######
                padded_img[ row_ind_1:row_ind_2 ,col_ind_1:col_ind_2 ] = IDCT

    # TODO: Remove out-of-range values
    ###### Your code here ######
    #将越界像素值设为边界值
    padded_img[padded_img < 0] = 0
    padded_img[padded_img > 255] = 255

    plt.imshow(np.uint8(padded_img),cmap='gray')
    plt.title("decoded paded image")
    plt.axis('off')
    plt.show()

    # TODO: Get original sized image from padded_img
    ###### Your code here ######
    decoded_img = padded_img[0:h, 0:w]

    plt.imshow(np.uint8(decoded_img),cmap='gray')
    plt.title("decoded image")
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    part1_encoder()
    part2_decoder()