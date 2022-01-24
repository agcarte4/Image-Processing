# Alex Carter 
# ECE 558 
# Project 1
# = = = = = = = = = = = = = = = =
'''
Problem 1:
    a)  conv2(img, filter, pad_type) - Performs 2d convolution using custom convolution function
        - Filters are of Kernel class, made constants at top
        - pad type: 'clip', 'wrap around', 'copy edge', 'reflect'
        --> Please see generate_filtered_images()
    b) Please see unit_impulse_test()

Problem 2:
    a) Please see DFT2(img)
        - Uses fft(seq, n= None) to perform 1D FFT using NumPy's FFT function
        - Please see generate_fft(Show= True) to see 2DFT of lena and wolves
    b) Please see IDFT2()
        - Uses DFT2() to perform DFT of array in frequency in domain
        - Please see compare_fft_idft() for comparison of the two

Notes:
    I wasn't sure how to organize the code. I would like better guidelines or a rubic for future project.
    The wolves.png image doesn't work with my conv2() implementation due to the shape of the image (ran out of time to figure it out).
'''
# = = = = = = = = = = = = = = = =


# Imports
import cv2 as cv
from matplotlib import scale
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.pyplot import imshow
from numpy import float32, ndarray
from numpy.core.fromnumeric import shape
from numpy.lib.arraypad import pad

# Simple filter class to make custom kernals
class Kernel:
    def __init__(self, scalar, matrix) -> None:
        self.scalar = scalar
        self.matrix = matrix
        self.rows = len(matrix)
        self.cols = len(matrix[0])
        
        # Filter Anchor point
        self.anchor_x = self.cols - int(self.cols/2) - 1
        self.anchor_y = self.rows - int(self.rows/2) - 1
        self.anchor = (self.anchor_x, self.anchor_y)

        # Filter Padding information
        self.x_pad_l = self.anchor_x
        self.x_pad_r = int(self.cols - self.anchor_y - 1)
        self.y_pad_t = self.anchor_y
        self.y_pad_b = int(self.rows - self.anchor_y - 1)

        self.pad = ((self.y_pad_t, self.y_pad_b), (self.x_pad_l, self.x_pad_r))

    # Print function for debugging 
    def __str__(self) -> str:
        s = str(self.scalar)
        m = str(self.matrix) 
        out = "Scalar = " + s + "\nMatrix = \n" + m
        return out


'''
Constants
'''
_debug = True
_print_debug = False

# # # Filter Constants
# 3x3 identity filter
ID_FILTER_3 = Kernel(scalar= 1, matrix= np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype= int))
# 3x3 box filter
BOX_3 = Kernel(scalar= (1/9), matrix= np.ones((3, 3), dtype= int))
# 2x1 first order derivative filters
FO_DX = Kernel(scalar= 1, matrix= np.array([-1, 1], dtype= int, ndmin=2))
FO_DY = Kernel(scalar= 1, matrix= np.array([[-1], [1]], dtype= int, ndmin=2))
# Prewitt Operators
PREWITT_X = Kernel(scalar= 1, matrix= np.array([[-1, 0, 1], [-1, 0, 1], [1, 0, -1]], dtype= int))
PREWITT_Y = Kernel(scalar= 1, matrix= np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype= int))
# Sobel Operators
SOBEL_X = Kernel(scalar= 1, matrix= np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype= int))
SOBEL_Y = Kernel(scalar= 1, matrix= np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype= int))
# Roberts Operators
ROBERTS_X = Kernel(scalar= 1, matrix= np.array([[0, 1], [-1, 0]], dtype= int))
ROBERTS_Y = Kernel(scalar= 1, matrix= np.array([[1, 0], [0, -1]], dtype= int))
# # # Additional 
# 3x3 sharpening filter
SHARP_3 = Kernel(scalar= 1, matrix= np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype= int))
# Gaussian Blur
GAUSS_BLUR_3 = Kernel(scalar= (1/16), matrix= np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype= int))
GAUSS_BLUR_5 = Kernel(scalar= (1/256), matrix= np.array([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]], dtype= int))


'''
Functions
'''
# Show Image
def show(img):
    cv.imshow('Image', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

# Function for performing Convolution of 2D arrays
def conv(f, w):
    '''
    This function only works with 2D arrays
    Inputs:
    f - image section (NxM array)
    w - kernel class  (scalar, NxM array)
    
    Outputs:
    g - single value result of convolution
    
    filtered_image[x, y] = w * img(x, y) = sum(dx = -a, a){ sum(dy = -b, b){ w[dx, dy]*f[x+dx, y+dy]  }  }

    '''
    # This variable is the index change to from center to to corners
    #   -1,-1   0,-1    1,-1
    #   -1,0    0,0     1,0
    #   -1,1    0,1     1,1   

    a_range = range(0, len(f))
    b_range = range(0, len(f[0]))

    if _print_debug:
        print('image section: ')
        print(f)
        print(w)

    # Internal - Ensure the filter and section have same size
    assert ((len(f) == w.rows) and (len(f[0]) == w.cols)), 'Image section must be the same size as the filter'

    sum = 0
    for s in a_range:
        for t in b_range:
            sum = sum + w.matrix[s, t] * f [s, t]
    sum = int(w.scalar * sum)
    
    if _debug:
        # Another method: use matrix multiplication (matrices need to be flattened)
        f = f.flatten()
        m = w.matrix.flatten()
        sum1 = np.matmul(m, f)
        sum1 = int(w.scalar * sum1)

        assert sum1 == sum, 'two sums are not equal'

    return sum

# This function takes an image and adds padding
def pad_image(img, pad_type= None, pad_size= None) -> ndarray:
    mode = ''

    # # None
    if pad_type is None:
        return img
    
    # # Zero Padding
    elif pad_type == 'clip' or pad_type == 'zerp-padding' or pad_type == 'clip/zero-padding':
        mode = 'constant'

    # # Wrap Around
    elif pad_type == 'wrap around':
        mode = 'wrap'

    # # Copy Edge
    elif pad_type == 'copy edge':
        mode = 'edge'

    # # Reflect at Edges
    elif pad_type == 'reflect' or pad_type == 'reflect across edge':
        mode = 'reflect'
        
    try:
        return np.pad(img, pad_size, mode) 

    except ValueError as err:
        raise ValueError ("Improper padding type provided, please use one of the following: \n\t1 - clip, zero-padding, clip/zero-padding \n\t2 - wrap around \n\t3 - copy edge \n\t4 - reflect, reflect across edge")


# Function to perfrom 2D convolution across whole image
def conv2(img, w, pad = None):
    # add stride later?

    # Since Python passes parameters by reference, I discovered that I needed to make a copy of the image or 
    #   the original image would be changed if this function was repeatedly called.
    # As a result of having to do this, this function is about twice as slow
    copy = np.zeros(img.shape)
    np.copyto(copy,img)

    # Is this image rgb?
    is_grey = (len(copy.shape) == 2) 

    rows = len(copy)
    cols = len(copy[0])

    # # # Need to be able to deal with even sized kernels! 
    # Pad the image based on the selected type of padding

    if _print_debug:
        p_rows = int(rows + np.floor(0.5 * w.rows))
        p_cols = int(cols + np.floor(0.5 * w.cols))
        print('image size: ', rows, cols)
        print('filter size: ', w.rows, w.cols)
        print('padded image size: ', p_rows, p_cols)
    
    if is_grey:
        p_img = pad_image(img= copy, pad_type= pad, pad_size= w.pad)
    elif not is_grey:
        # pad rgb image
        # Split into individual channels to perform 2D convolution
        b_p_img = pad_image(img= copy[:,:,0], pad_type= pad, pad_size= w.pad)
        g_p_img = pad_image(img= copy[:,:,1], pad_type= pad, pad_size= w.pad)
        r_p_img = pad_image(img= copy[:,:,2], pad_type= pad, pad_size= w.pad)
        
        # # Combine channels
        p_img = np.stack(arrays= (b_p_img, g_p_img, r_p_img), axis=2)
    
    #show(p_img)

    # # Perform Convolution
    # ndarrays are indexed by row then column
    if is_grey:
        for r in range(w.x_pad_l, rows ):
            for c in range(w.y_pad_t, cols ):
                # Perform convultion around each pixel
                xl = r-w.x_pad_l
                xr = r+w.x_pad_r+1
                ct = c-w.y_pad_t
                cb = c+w.y_pad_b+1
                
                sec = p_img[ct: cb, xl: xr]
                copy[ct,xl] = conv(f= sec, w= w) 

    elif not is_grey:
        for r in range(w.x_pad_l, rows ):
            for c in range(w.y_pad_t, cols):
                # Perform convultion around each pixel in each channel
                xl = r-w.x_pad_l
                xr = r+w.x_pad_r+1
                ct = c-w.y_pad_t
                cb = c+w.y_pad_b+1

                # OpenCV uses BGR, matplotlib uses RGB 
                b_sec = p_img[ct: cb, xl: xr, 0]
                copy[ct,xl, 0] = conv(f= b_sec, w= w)
                g_sec = p_img[ct: cb, xl: xr, 1]
                copy[ct,xl, 1] = conv(f= g_sec, w= w)
                r_sec = p_img[ct: cb, xl: xr, 2]
                copy[ct,xl, 2] = conv(f= r_sec, w= w)

    return copy

def generate_filtered_images():
    ### the wolves image does not work...

    # # Pad types: 
    # 1 - clip, zero-padding, clip/zero-padding
    # 2 - wrap around
    # 3 - copy edge 
    # 4 - reflect, reflect across edge

    # # # Filters
    # ID_FILTER_3   - 3x3 Identity Filter
    # BOX_3         - 3x3 Box Filter
    # FO_DX         - 2x1 First-order X Derivative Filter
    # FO_DY         - 1x2 First-order Y Derivative Filter
    # PREWITT_X     - 3x3 Prewitt X Filter
    # PREWITT_Y     - 3x3 Prewitt Y Filter
    # SOBEL_X       - 3x3 Sobel X Filter
    # SOBEL_Y       - 3x3 Sobel Y Filter
    # ROBERTS_X     - 2x2 Roberts X Filter
    # ROBERTS_Y     - 2x2 Roberts Y Filter
    # SHARP_3       - 3x3 Sharpening Filter
    # GAUSS_BLUR_3  - 3x3 Gaussian Blurring Filter
    # GAUSS_BLUR_5  - 5x5 Gaussian Blurring Filter


    '''
    w = SOBEL_X
    i = cv.imread("C:/Users/bravo/Desktop/558/proj1/lena.png", cv.IMREAD_GRAYSCALE)
    i = conv2(i, w, pad = 'clip')

    # These two produce significantly different images!
    show(i)
    plt.show(imshow(i, cmap='gray'))
    '''

    all_filters = [ ID_FILTER_3, BOX_3, FO_DX, FO_DY, PREWITT_X, PREWITT_Y, SOBEL_X, SOBEL_Y, ROBERTS_X, ROBERTS_Y]
    fil_names = ['ID_FILTER_3', 'BOX_3', 'FO_DX', 'FO_DY', 'PREWITT_X', 'PREWITT_Y', 'SOBEL_X', 'SOBEL_Y', 'ROBERTS_X', 'ROBERTS_Y']
    all_pads = ['clip', 'wrap around', 'copy edge', 'reflect']

    lena_rgb = cv.imread("C:/Users/bravo/Desktop/558/proj1/lena.png")
    lena_grey = cv.imread("C:/Users/bravo/Desktop/558/proj1/lena.png", cv.IMREAD_GRAYSCALE)
    wolves_rgb = cv.imread("C:/Users/bravo/Desktop/558/proj1/wolves.png")
    wolves_grey = cv.imread("C:/Users/bravo/Desktop/558/proj1/wolves.png", cv.IMREAD_GRAYSCALE)

    # Iterate all prossible combinations
    for p in range(0, len(all_pads)):
        for w in range(0, len(all_filters)):
            # Lena Grey
            new = conv2(lena_grey, w=all_filters[w], pad=all_pads[p])

            file_name = 'C:/Users/bravo/Desktop/558/proj1/lena_grey_' + str(all_pads[p] +'_'+str(fil_names[w])+'.png')
            cv.imwrite(filename= file_name, img= new)

            # Lena RGB
            new = conv2(lena_rgb, w=all_filters[w], pad=all_pads[p])

            file_name = 'C:/Users/bravo/Desktop/558/proj1/lena_rgb_' + str(all_pads[p] +'_'+str(fil_names[w])+'.png')
            cv.imwrite(filename= file_name, img= new)
            
            '''
            # # Not working?
            # wolves Grey
            new = conv2(wolves_grey, w=all_filters[w], pad=all_pads[p])

            file_name = 'C:/Users/bravo/Desktop/558/proj1/wolves_grey_' + str(all_pads[p] +'_'+str(fil_names[w])+'.png')
            plt.imsave(fname=file_name, arr=new, cmap='gray')

            # Wolves RGB
            new = conv2(wolves_rgb, w=all_filters[w], pad=all_pads[p])

            file_name = 'C:/Users/bravo/Desktop/558/proj1/wolves_rgb_' + str(all_pads[p] +'_'+str(fil_names[w])+'.png')
            plt.imsave(fname=file_name, arr=new)
            '''


def unit_impulse_image(shape = (1024,1024)) -> ndarray:
    unit_impulse = np.zeros(shape)

    x_cen = int(len(unit_impulse)/2)
    y_cen = int(len(unit_impulse[0])/2)

    unit_impulse[x_cen, y_cen] = 255
    return unit_impulse

def unit_impulse_conv2(impulse, w) -> ndarray:
    out = conv2(impulse, w, pad='clip')
    cv.imwrite(filename='C:/Users/bravo/Desktop/558/proj1/unit_impulse_conv.png', img= out)
    return out

def unit_impulse_test() -> None:
    unit = unit_impulse_image()
    result = unit_impulse_conv2(unit, w=GAUSS_BLUR_5)
    plt.show(imshow(result, cmap='gray'))


def histogram_equlization(img):
    # histogram equalization
    # get the value counts
    row_count = len(img)
    col_count = len(img[0])

    scaled_img = np.asarray(img, dtype=float32 )

    keys = [*range(0,256)]
    pixel_counts = {key: None for key in keys}
    for r in range(0, row_count):
        for c in range(0, col_count):
            v = scaled_img[r, c]
            if pixel_counts[v] is None:
                pixel_counts[v] = 1
            else:
                pixel_counts[v] = pixel_counts[v] + 1

    # remove Nones from dict
    for x in keys:
        if pixel_counts[x] is None:
            pixel_counts.pop(x)

    # store CDF
    cdf = 0
    for x in pixel_counts.keys():
        cdf = pixel_counts[x] + cdf
        pixel_counts[x] = cdf


    # the values needed for equalization
    cdf_min = min(pixel_counts)
    MN = row_count * col_count
    L = 255

    # perform equalization
    for r in range(0,row_count):
        for c in range(0, col_count):
            old_v = scaled_img[r, c]
            old_v = pixel_counts[old_v] - cdf_min
            new_v = old_v / (MN - cdf_min)
            #new_v = new_v * L
            scaled_img[r, c] = new_v

    return scaled_img

def fft(seq, n= None) -> ndarray:
    # Using NumPy's built-in FFT routine
    #np.fft.fft()
    return np.fft.fft(seq)

def DFT2(img) -> ndarray:
    fft_img = np.zeros(img.shape, dtype=complex)

    # Perfrom 1d DFT of each row 
    for r in range(0, len(img)):
        fft_img[r,:] = fft(img[r, :])
    
    # Perfrom 1d DFT on each column using previus results
    for c in range(0, len(img[0])):
        fft_img[:,c] = fft(fft_img[:,c])     
    
    return fft_img


def generate_fft(show=True):
    # Problem 2a

    lena_grey = cv.imread("C:/Users/bravo/Desktop/558/proj1/lena.png", cv.IMREAD_GRAYSCALE)

    # Scale the image to be between 0...1
    myLena = histogram_equlization(lena_grey)
    myLena = DFT2(myLena)
    #myLena_a = np.log(np.abs(np.fft.fftshift(myLena))**2)
    # Amplitude
    myLena_a = np.log(1+np.abs(np.fft.fftshift(myLena)))
    if show:
        plt.show(imshow(myLena_a, cmap='gray'))
    # Angle
    myLena_p = np.angle(np.fft.fftshift(myLena))
    if show:
        plt.show(imshow(myLena_p, cmap='gray'))

    # Using NumPy's 2D FFT to compare
    ftlena = np.fft.fft2(lena_grey)
    ftlena_a = np.log(np.abs(np.fft.fftshift(ftlena))**2)
    if show:
        plt.show(imshow(ftlena_a, cmap='gray'))
    # Angle
    ftlena_p = np.angle(np.fft.fftshift(ftlena))
    if show:
        plt.show(imshow(ftlena_p, cmap='gray'))

    # Wolves
    wolves_grey = cv.imread("C:/Users/bravo/Desktop/558/proj1/wolves.png", cv.IMREAD_GRAYSCALE)

    # Scale the image to be between 0...1
    myWolves = histogram_equlization(wolves_grey)
    myWolves = DFT2(myWolves)
    # scaled = np.log(np.abs(np.fft.fftshift(scaled))**2)
    # Amplitude
    myWolves_a = np.log(1+np.abs(np.fft.fftshift(myWolves)))
    if show:
        plt.show(imshow(myWolves_a, cmap='gray'))
    # Angle
    myWolves_p = np.angle(np.fft.fftshift(myWolves))
    if show:
        plt.show(imshow(myWolves_p, cmap='gray'))

    # Using NumPy's 2D FFT to compare
    ftwolves = np.fft.fft2(wolves_grey)
    ftwolves_a = np.log(np.abs(np.fft.fftshift(ftwolves))**2)
    if show:
        plt.show(imshow(ftwolves_a, cmap='gray'))
    # Angle
    ftwolves_p = np.angle(np.fft.fftshift(ftwolves))
    if show:
        plt.show(imshow(ftwolves_p, cmap='gray'))

    return myLena, myWolves

def IDFT2() -> ndarray:
    lena_fft, = generate_fft(False)
    lena_ifft = (1 / len(lena_fft)) * (DFT2(lena_fft))
    return np.rot90(np.rot90(np.log(1+np.abs(lena_ifft))))

def compare_fft_ifft() -> None:
    # True image
    lena_grey = cv.imread("C:/Users/bravo/Desktop/558/proj1/lena.png", cv.IMREAD_GRAYSCALE)
    lena_ifft = IDFT2()

    plt.show(imshow(lena_ifft, cmap='gray'))

    plt.show(imshow(lena_grey-lena_ifft, cmap='gray'))

'''
Call functions here
'''



