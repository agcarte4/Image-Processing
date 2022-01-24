# Alex Carter 
# ECE 558 
# Project 2
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
from numpy import dtype, float32, ndarray, zeros
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

'''
The code below is adapted from my senior design project.
I changed many aspects but much of the core is the same.
This GUI allows the user to load in a foreground and background image, 
compute the Gaussian and Laplacian pyramids, and then create a mask for 
Laplacian pyramid blending. The user also has access to several functions 
to save the results.

- Does not handle different sized images, assumes images are of the same size
- no RGB, only grayscale
'''
# # # # #
# GUI
class GUI:
    def __init__(self, window, title, layers = None):
        ## Creating a window
        self.title = title
        self.window = window 
        self.window.option_add('*tearOff', False)
        self.window.title(title)

        # layer count
        if layers:
            self.layers = layers
        else: 
            self.layers = 20 # max?

        # iamges
        self.fore = None
        self.back = None
        self.mask = None
        self.blend_img = None

        # region selected for mask (rectangle)
        self.r = None

        # pyramids for foreground
        self.fgpyr = None
        self.flpyr = None
        self.fgcomp = None
        self.flcomp = None

        # pyramids for background
        self.bgpyr = None
        self.blpyr = None
        self.bgcomp = None
        self.blcomp = None

        # pyramids for mask region
        self.mgpyr = None
        self.mlpyr = None

        # controls
        self.fore_change = False
        self.back_change = False
        self.mask_change = False
        self.blend_change = False

        ##############################################
        #                   MENUS                    #
        ##############################################

        ## Main menu bar
        self.menu          = tk.Menu(self.window)           
        self.file_menu     = tk.Menu(self.menu)             # File sub menu child
        
        ## Items for file submenu
        self.file_menu.add_command(label = 'New Foreground Image', command=self.new_fore)
        self.file_menu.add_command(label = 'New Background Image', command=self.new_back)
        self.file_menu.add_command(label = 'Save Blended...', command=self.save)
        self.file_menu.add_command(label = 'Clear Canvas', command=self.clear_canvas)
        self.file_menu.add_command(label = 'Exit', command=self.exit)
        self.menu.add_cascade(label='File', menu=self.file_menu)

        self.window.config(menu=self.menu)

        ##  Sets up space for the canvas
        self.canvas = tk.Canvas(window, width=500, height=500)
        self.canvas.grid(column = 0, row = 0, columnspan = 8, rowspan = 8)
        
        ##############################################
        #                  BUTTONS                   #
        ##############################################
        self.fore_pyr_button = tk.Button(window, text = "Compute Foreground Pyramids", command=self.compute_pyr_fore)
        self.fore_pyr_button.grid(column = 0, row = 8, sticky = "nsew", padx = 10, pady= 5)
        
        self.back_pyr_button = tk.Button(window, text = "Compute Background Pyramids", command=self.compute_pyr_back)
        self.back_pyr_button.grid(column = 1, row = 8, sticky = "nsew", padx = 10, pady= 5)
        
        self.square_mask = tk.Button(window, text = "Square mask", command=self.draw_square_mask)
        self.square_mask.grid(column = 3, row = 8, sticky = "nsew", padx = 10, pady= 5)
    
        self.ellipse_mask = tk.Button(window, text = "Ellipse mask", command=self.draw_ellipse_mask)
        self.ellipse_mask.grid(column = 4, row = 8, sticky = "nsew", padx = 10, pady= 5)

        self.mask_pyr_btn = tk.Button(window, text = "Compute Mask Pyramids", command= self.compute_pyr_mask)
        self.mask_pyr_btn.grid(column = 5, row = 8, sticky = "nsew", padx = 10, pady= 5)

        self.blend_btn = tk.Button(window, text = "Blend", command=self.blend)
        self.blend_btn.grid(column = 6, row = 8, sticky = "nsew", padx = 10, pady= 5)     

        self.save_btn = tk.Button(window, text = "Save Blend", command=self.save)
        self.save_btn.grid(column = 8, row = 8, sticky = "nsew", padx = 10, pady= 5)        

        ## Safe extit variable and protocol
        self.quit = False
        self.window.protocol("WM_DELTE_WINDOW", self.exit)

        ## Initialize canvas
        self.update()

        ## Start the main tkinter loop
        self.window.mainloop()

    def update(self):
        #self.canvas.delete("all")

        # Create the image window on the GUI, and update it
        if self.fore_change:
            # update window to accomodate image
            self.canvas.config(width= self.fore.shape[1], height=self.fore.shape[0])

            self.img1 = ImageTk.PhotoImage(image=Image.fromarray(self.fore))
            self.canvas.create_image(0, 0, image=self.img1, anchor=tk.NW)
        
        if self.back_change:
            self.canvas.config(width= self.fore.shape[1]+self.back.shape[1], height=self.fore.shape[0])

            self.img2 = ImageTk.PhotoImage(image=Image.fromarray(self.back))
            self.canvas.create_image(self.fore.shape[1], 0, image=self.img2, anchor=tk.NW)
        
        if self.blend_change:
            self.canvas.config(width= self.fore.shape[1]+self.back.shape[1], height=2*self.fore.shape[0])

            self.img3 = ImageTk.PhotoImage(image=Image.fromarray(self.blend_img))
            self.canvas.create_image(int(0.5*self.fore.shape[1]), self.fore.shape[0], image=self.img3, anchor=tk.NW)
            self.blend_change = False

        # exit 
        if self.quit:
            exit()
        else:
            self.job_id = self.canvas.after((100), self.update)
        
    def clear_canvas(self):
        self.canvas.delete("all")

    def new_image(self):
        fileName = tk.filedialog.askopenfilename(title = "Open an image.. ",filetypes =[("Image files","*.png")])
        return fileName

    def new_fore(self):
        file = self.new_image()
        self.fore = cv.imread(file, cv.IMREAD_GRAYSCALE)
        self.fore_change = True

    def new_back(self):
        file = self.new_image()
        self.back = cv.imread(file, cv.IMREAD_GRAYSCALE)
        self.back_change = True

    def compute_pyr_fore(self):
        if self.fore is None:
            # warn user to select foreground image
            tk.messagebox.showerror('Error', 'Please select a foreground image.')
            return

        if self.fore_change is False:
            # no reason to recompute
            tk.messagebox.showinfo('Information', 'No reason to recompute.')
            return

        self.window.title('Computing Foreground Image Pyramid....')
        tk.messagebox.showinfo('Information', 'Please wait while the foreground pyramids are computed. This may take a few minutes.')
        self.fgpyr, self.flpyr = ComputePyr(self.fore, self.layers)
        
        self.fgcomp = pyramidImage(self.fgpyr)
        self.flcomp = pyramidImage(self.flpyr)
        self.fore_change = False
        self.window.title(self.title)
    
    def compute_pyr_back(self):
        if self.back is None:
            # warn user to select background image
            tk.messagebox.showerror('Error', 'Please select a background iamge.')
            return

        if self.back_change is False:
            # no reason to recompute
            tk.messagebox.showinfo('Information', 'No reason to recompute.')
            return

        self.window.title('Computing Background Image Pyramid....')
        tk.messagebox.showinfo('Information', 'Please wait while the background pyramids are computed. This may take a few minutes.')
        self.bgpyr, self.blpyr = ComputePyr(self.back, self.layers)

        self.bgcomp = pyramidImage(self.bgpyr)
        self.blcomp = pyramidImage(self.blpyr)
        self.back_change = False
        self.window.title(self.title)

    def compute_pyr_mask(self):
        if self.mask is None:
            # warn user to select background image
            tk.messagebox.showerror('Error', 'Please create a mask first.')
            return

        if self.mask_change is False:
            # no reason to recompute
            tk.messagebox.showinfo('Information', 'No reason to recompute.')
            return

        self.window.title('Computing Mask Image Pyramid....')
        tk.messagebox.showinfo('Information', 'Please wait while the mask pyramids are computed. This may take a few minutes.')
        self.mgpyr, self.mlpyr = ComputePyr(self.mask, self.layers)

        self.mask_change = False
        self.window.title(self.title)   

    def draw_square_mask(self):
        self.mask = np.zeros(self.fore.shape[:2], dtype='uint8')
        r = cv.selectROI("Please select ROI", self.fore)
        cv.destroyAllWindows()

        # create mask
        self.r = r
        self.mask[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = 1

        cv.imshow("Mask", np.bitwise_and(self.fore, 255*self.mask))
        cv.waitKey(0)

        self.mask_change = True

    def draw_ellipse_mask(self):
        print('ellipse')
        self.mask = np.zeros(self.fore.shape[:2], dtype='uint8')

        r = cv.selectROI("Please select ROI", self.fore)
        cv.destroyAllWindows()

        self.r = r
        # r[1] - y0
        # r[3] - offset from y0
        # r[0] - x0
        # r[2] - offsect from x0
        axis_x = int(0.5*r[2])
        axis_y = int(0.5*r[3])
        cen_x = r[0] + axis_x
        cen_y = r[1] + axis_y
        center =(cen_x, cen_y)
        axes = (axis_x, axis_y)
    
        # needs to be updated for RGB
        color = 1

        self.mask = cv.ellipse(self.mask, center=center, axes = axes, angle=0, startAngle=0, endAngle=360, color=color, thickness=-1)
    
        cv.imshow("Mask", np.bitwise_and(self.fore, 255*self.mask))
        cv.waitKey(0)

        self.mask_change = True

    def blend(self):
        # compute any pyramids that may be missing
        if self.fore_change:
            self.compute_pyr_fore()
        if self.back_change:
            self.compute_pyr_back()
        if self.mask_change or self.mask is None:
            self.compute_pyr_mask()
        
        # select center point on background image to apply mask
        #self.back_cen = (0,0)

        self.window.title("Blending in progress...")
        tk.messagebox.showinfo('Information', 'Please wait while the blending is computed. This may take a few minutes.')

        # check that all pyramids have the same num of layers
        assert len(self.flpyr) == len(self.blpyr)
        assert len(self.flpyr) == len(self.mgpyr)
        
        # first combine the laplacian pyramids and gaussian mask pyramids
        lap = list()
        for l in range(0, len(self.flpyr)):
            mask = self.mgpyr[l]
            f = self.flpyr[l] * mask

            mask = 1 - mask           
            b = mask * self.blpyr[l]

            fb = f+b
            lap.append(fb)
        

        # collapse the result pyramid
        # start with top layer
        res = lap[len(lap)-1]
        for l in range(len(lap)-2, -1, -1):
            # upscale previous res
            size = (lap[l].shape[1], lap[l].shape[0])
            up = cv.resize(res, dsize=size, fx=0, fy=0, interpolation=cv.INTER_NEAREST)
            # blur to remove artifacts
            up = conv2(up, GAUSS_BLUR_5, 'reflect')

            # add current layer to upscaled result of previous
            res = cv.add(lap[l],up)
            #plt.show(imshow(res, cmap='gray'))



        self.window.title(self.title)
        self.blend_img = res
        self.blend_change = True

        #plt.show(imshow(self.flcomp, cmap='gray'))
        #plt.show(imshow(self.blcomp, cmap='gray'))
        #plt.show(imshow(self.mask, cmap='gray'))

    def safe_exit(self):
        self.quit = True
        self.window.after_cancel(self.job_id)

    def exit(self):
        # Cancel all future canvas updates
        self.window.destroy()
    
    # Open fileDialog so user can select location and name for file
    def save(self):
        if self.blend_img is None:
            tk.messagebox.showerror('Error', 'There is no blended image to save!')
            return False

        fileName = tk.filedialog.asksaveasfilename(title = "Save As",filetypes =[("Image files","*.png")])
        if (fileName is not ()) and (fileName is not ''):
            try:
                plt.imsave(fname=fileName, arr=self.blend_img, cmap='gray')
            except FileExistsError:
                print('File already exists!') 
            return True
        return False

# # #  # # #
#   Main   #
# # #  # # #

'''
im1 = cv.imread("C:/Users/bravo/Desktop/558/agcarte4_project02/shark.png", cv.IMREAD_GRAYSCALE)
im2 = cv.imread("C:/Users/bravo/Desktop/558/agcarte4_project02/earth.png", cv.IMREAD_GRAYSCALE)
im3 = cv.imread("C:/Users/bravo/Desktop/558/agcarte4_project02/ab.png", cv.IMREAD_GRAYSCALE)
im4 = cv.imread("C:/Users/bravo/Desktop/558/agcarte4_project02/shrek.png", cv.IMREAD_GRAYSCALE)
im5 = cv.imread("C:/Users/bravo/Desktop/558/agcarte4_project02/pool.png", cv.IMREAD_GRAYSCALE)
#plt.show(imshow(wolves, cmap='gray'))
#wolves_2 = conv2(wolves, GAUSS_BLUR_5, 'reflect')
#plt.show(imshow(wolves_2, cmap='gray'))

for i in [im1, im2, im3, im4, im5]:
    gPyr, lPyr = ComputePyr(i)
    gPyrImg = pyramidImage(gPyr)
    lPyrImg = pyramidImage(lPyr)
    plt.show(imshow(gPyrImg, cmap='gray'))
    plt.show(imshow(lPyrImg, cmap='gray'))

'''

master = GUI(tk.Tk(), "Project 2")
