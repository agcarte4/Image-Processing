# Alex Carter 
# ECE 558 
# Project 3
# = = = = = = = = = = = = = = = =
'''
Problem 1:
    a) ComputePyr(img, num_layers) - Computes the Gaussian and Laplacian pyramids
    b) GUI - Used to create a mask: open image, select region for mask
    c) LaplaceBlend() - Computes a blended image uses two source images and the Laplacian pyramid?
Notes:
    
'''
# = = = = = = = = = = = = = = = =


# Imports
import cv2 as cv
from matplotlib import cm, scale
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.pyplot import imshow, summer, title
from numpy import double, dtype, float16, float32, longfloat, ndarray, ulonglong, zeros
from numpy.core.fromnumeric import shape
from numpy.lib.arraypad import pad

# for gui
import tkinter as tk
from tkinter import *
from PIL import ImageTk, Image
import os as os


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


# # # Filter Constants
# 3x3 identity filter
ID_FILTER_3 = Kernel(scalar= 1, matrix= np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype= int))
# Gaussian Blur
GAUSS_BLUR_3 = Kernel(scalar= (1/16), matrix= np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype= int))
GAUSS_BLUR_5 = Kernel(scalar= (1/256), matrix= np.array([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]], dtype= int))
# Laplacian Filter
LAPLACE_1 = Kernel(scalar= 1, matrix= np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype= int))
LAPLACE_2 = Kernel(scalar= 1, matrix= np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype= int))


'''
Constants
'''
_debug = True
_print_debug = False



### From Project 1 ###
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
        for sigma in b_range:
            sum = sum + w.matrix[s, sigma] * f [s, sigma]
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


def show(img):
    cv.imshow('Image', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

# Function to perfrom 2D convolution across whole image
def conv2(img, w, pad = None):
    # add stride later?

    # Since Python passes parameters by reference, I discovered that I needed to make a copy of the image or 
    #   the original image would be changed if this function was repeatedly called.
    # As a result of having to do this, this function is about twice as slow
    
    
    # copy = np.zeros(img.shape)
    # np.copyto(copy,img)
    copy = img.copy()

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
    
    # show(p_img)

    # # Perform Convolution
    # ndarrays are indexed by row then column
    if is_grey:
        for r in range(w.x_pad_l, rows ):
            for c in range(w.y_pad_t, cols ):
                # Perform convultion around each pixel
                xl = c-w.x_pad_l
                xr = c+w.x_pad_r+1
                ct = r-w.y_pad_t
                cb = r+w.y_pad_b+1
                
                sec = p_img[ct: cb, xl: xr]

                copy[ct,xl] = conv(f= sec, w= w)

    elif not is_grey:
        for r in range(w.x_pad_l, rows ):
            for c in range(w.y_pad_t, cols):
                # Perform convultion around each pixel in each channel
                xl = c-w.x_pad_l
                xr = c+w.x_pad_r+1
                ct = r-w.y_pad_t
                cb = r+w.y_pad_b+1

                # OpenCV uses BGR, matplotlib uses RGB 
                b_sec = p_img[ct: cb, xl: xr, 0]
                copy[ct,xl, 0] = conv(f= b_sec, w= w)
                g_sec = p_img[ct: cb, xl: xr, 1]
                copy[ct,xl, 1] = conv(f= g_sec, w= w)
                r_sec = p_img[ct: cb, xl: xr, 2]
                copy[ct,xl, 2] = conv(f= r_sec, w= w)

    # show(p_img)
    return copy
### ###
def pyramidImage(pyr) -> ndarray:
    # shape[0] = y - num of rows
    # shape[1] = x - num of cols

    # create newe image large enough for all layers
    base_size = (pyr[0].shape[0], pyr[0].shape[1]+pyr[1].shape[1])
    comp_image = np.zeros(base_size)

    prev_cols = 0
    prev_rows = 0
    for l in range(0, len(pyr)):
        #print(l, pyr[l].shape)
        # get size of layer
        cur_rows = pyr[l].shape[0]
        cur_cols = pyr[l].shape[1]
        comp_image[prev_rows: prev_rows + cur_rows, prev_cols: prev_cols + cur_cols] = pyr[l]
        
        # update previous layers info
        if l is 0:
            prev_cols = cur_cols
        else:
            prev_rows = cur_rows + prev_rows
    # return single image with all layers
    return comp_image


def ComputePyr(img, num_layers = 20) -> ndarray:
    '''
    This functions computes the Gaussian and Laplacian pyramids of a provided input image.
    Inputs:
        img - is an input image (grey, or RGB)
        num_layers - is the number of layers of the pyramid to be computed. Depending on the size, needs to be checked if valid. 
            If not, use the maximum value allowed in terms of the size of the input image.
    Outputs: 
        gPyr - Gaussian pyramid 
        lPyr - Laplacian pyramid
    '''

    # Compute Gaussian images
    gaus_list = list()

    # append original img array to python list
    gaus_list.append(img)
    
    # Iterate for the specified number of layers
    for l in range(0, num_layers):
        # ADD RGB SUPPORT

        # get previous layer image
        new_img = gaus_list[l]

        # Check dimensions of next layer if needed 
        # If current layer has dimensions less than 1, then the pyramid is done
        if ((new_img.shape[0] is 1) or (new_img.shape[1] is 1)):
            break

        # apply smoothing to previous layer image
        smoothed = conv2(new_img, GAUSS_BLUR_3, 'reflect')

        if _print_debug:
            print('size after blurring = ', smoothed.shape)

        # down sample the smoothed image
        down_samp = cv.resize(smoothed, dsize= None, fx=0.5, fy=0.5, interpolation= cv.INTER_NEAREST)

        if _print_debug:
            print('size after downsampling = ', down_samp.shape)
            plt.show(imshow(down_samp, cmap='gray'))

        # append to pyramid list
        gaus_list.append(down_samp)

    gPyr = np.asanyarray(gaus_list)

    # # # # #
    # compute laplacian pyramid
    lap_list = list()

    for l in range(0, len(gaus_list)-1):
        # get currnet gaussian layer
        gaus = gaus_list[l]

        # get next layer and upsample
        dsize = (gaus.shape[1], gaus.shape[0])
        up_gaus = cv.resize(gaus_list[l+1], dsize= dsize, fx= 0, fy= 0, interpolation= cv.INTER_NEAREST)

        # apply blurring
        smoothed = conv2(up_gaus, GAUSS_BLUR_5, 'reflect')

        # compute laplacian
        lap_img = gaus - smoothed

        # add new laplcian image to list
        lap_list.append(lap_img)


    
    # add last Gaussian to Laplacian 
    lap_list.append(gaus_list[l+1])

    lPyr = np.asanyarray(lap_list)

    # return the gaussian and laplacian pyramids
    return gPyr, lPyr

# Histogram equilization
def histogram_equlization(image):
    # histogram equalization
    # get the value counts
    he_image = image

    #he_image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    
    
    row_count = len(image)
    col_count = len(image[0])

    keys = [*range(0,256)]
    pixel_counts = {key: None for key in keys}
    for r in range(0,row_count):
        for c in range(0, col_count):
            v = he_image[r][c]
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
    L = 255 # assuming grayscale

    # perform equalization
    for r in range(0, row_count):
        for c in range(0, col_count):
            old_v = he_image[r][c]
            old_v = pixel_counts[old_v] - cdf_min
            old_v = old_v / (MN - cdf_min)
            new_v = old_v * L
            he_image[r][c] = round(new_v)
    
    return he_image


'''
Project 3 Specific

'''
# Create the LoG filter
def LoGmaker(s):# -> Kernel:
    # LoG filter maker
    # 3 std deviations from center
    sigma = s
    w = np.floor(3*sigma)

    x,y = np.ogrid[-w:w+1, -w:w+1]

    xy2 = x**2 + y**2
    exp = (-1*xy2)/(2*(sigma**2))
    
    f = (1/(np.pi*sigma**2))*(1-((xy2)/(2*sigma**2)))*(np.e**exp)  ## scaled log

    k = Kernel(scalar= 1, matrix= np.asarray(f))#, dtype=int))

    return k


# Create LoG at scale, filter, save square
def LoG_Convolve(img, n, k):
    log_res = list()

    for i in range(n):
        r = k**i
        s = r
        print(s)

        log = LoGmaker(s = s)
        log_1 = conv2(img, log, 'reflect')
        log_1 = np.square(log_1)
        log_res.append(log_1)

        cv.imwrite('C:/Users/bravo/Desktop/558/agcarte4_project03/1.png', log_1)
    return np.stack(log_res)

# Perform non-maximum suppression
def Thresh(LoG, t = 0.05, n = 4):
    blob_res = list()
    print(im1.shape)
    print(LoG.shape)

    for y in range(LoG.shape[0]):
        for x in range(LoG.shape[1]):
            p = LoG[:,y-n:y+n+1,x-n:x+n+1] # Slice of 26 from each scale
            np.amax
            print(p.shape)





# # #  # # #
#   Main   #
# # #  # # #


im1 = cv.imread("C:/Users/bravo/Desktop/558/agcarte4_project03/base_images/butterfly.png", cv.IMREAD_GRAYSCALE)
im2 = cv.imread("C:/Users/bravo/Desktop/558/agcarte4_project03/base_images/fishes.jpg", cv.IMREAD_GRAYSCALE)
im3 = cv.imread("C:/Users/bravo/Desktop/558/agcarte4_project03/base_images/sunflowers.jpg", cv.IMREAD_GRAYSCALE)
im4 = cv.imread("C:/Users/bravo/Desktop/558/agcarte4_project03/base_images/einstein.jpg", cv.IMREAD_GRAYSCALE)
im5 = cv.imread("C:/Users/bravo/Desktop/558/agcarte4_project03/base_images/imp.png", cv.IMREAD_GRAYSCALE)
#plt.show(imshow(wolves, cmap='gray'))
#wolves_2 = conv2(wolves, GAUSS_BLUR_5, 'reflect')
#plt.show(imshow(wolves_2, cmap='gray'))


# HE
im1 = histogram_equlization(im1)
# plt.show(imshow(im1, cmap='gray'))


octaves = 3


# LoG
log_res = LoG_Convolve(im1, 2, np.sqrt(2))
Thresh(log_res, 0.02, 5)


'''
for i in [im1, im2, im3, im4, im5]:
    gPyr, lPyr = ComputePyr(i)
    gPyrImg = pyramidImage(gPyr)
    lPyrImg = pyramidImage(lPyr)
    plt.show(imshow(gPyrImg, cmap='gray'))
    plt.show(imshow(lPyrImg, cmap='gray'))
'''


#master = GUI(tk.Tk(), "Project 2")
