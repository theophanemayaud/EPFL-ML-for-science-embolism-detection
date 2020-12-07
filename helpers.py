import glob, os, ntpath
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label as region_map
import cv2 as cv

def extract_name(path):
    '''
    A method to reove folder prefix and file type suffix from file full names
    Input:
    :path: The full path of the file including folders and file type
    Output:
    :name: The stripped down file name
    '''
    head, tail = ntpath.split(path)
    head = tail or ntpath.basename(head)
    return os.path.splitext(head)[0]

def find_best_sigma(img, range=100):
    '''
    A method to find the best estimation of the Gaussian noise of an image
    Input:
    :img: Grayscale image as numpy array
    :range: the distance to explore away from the corners
    Output:
    :sigma: estimated sigma of gaussian noise
    '''
    sigma_1 = np.std(img[:range,:range])
    sigma_2 = np.std(img[:range,-range:])
    sigma_3 = np.std(img[-range:,-range:])
    sigma_4 = np.std(img[-range:,:range])
    return np.min([sigma_1,sigma_2,sigma_3,sigma_4])

def find_otsu(img):
    '''
    A method to find the best threshold value according to otsu's method
    Input:
    :img: Grayscale image as numpy array
    Output:
    :t_otsu: estimated threshold value according to Otsu's method
    '''
    hist, bins = np.histogram(img.ravel(), np.arange(img.min(),img.max()+2,1),density=True)
    sigma = np.zeros_like(hist)
    for t in range(len(hist)):
        w0 = np.sum(hist[0:t])
        w1 = np.sum(hist[t:-1])
        mu0 = hist[:t]@bins[:t] / w0
        mu1 = hist[t:]@bins[t:-1] / w1
        sigma[t] = w0*w1*(mu0-mu1)**2
    return bins[np.argmax(np.nan_to_num(sigma))]

def remove_background(img, th=1000):
    '''
    A method to clear the background from the mask
    Input:
    :img: Grayscale image of the holes mask as numpy array
    Output:
    :img: the mask without to background
    '''
    regions, regions_num = region_map(img)
    for i in range(regions_num):
        if np.sum(regions==i)>= th:
            img[regions==i] = 0
    return img

def extend_mirror(img, out_size):
    '''
    A method to extend an image to certain resolution by mirrorring the edges
    Input:
    :img: image as numpy array
    :out_size: a tuple of the desired output resolution
    Output:
    :out: the extended image
    '''
    # input error exceptions
    if np.any(img.shape>tuple(out_size)):
        raise Exception('Error: at least on of out_size axes is smaller than the image shape')
    if np.any(3*img.shape>tuple(out_size)):
        raise Exception('Error: at least on of out_size axes is at least 3 times larger than the image shape')
    # output parameters
    out = np.zeros(out_size)
    v_edge_u = (out_size[0]-img.shape[0]) // 2
    v_edge_d = -(out_size[0]-img.shape[0]-v_edge_u)
    h_edge_l = (out_size[1]-img.shape[1]) // 2
    h_edge_r = -(out_size[1]-img.shape[1]-h_edge_l)
    # output centre
    out[v_edge_u:v_edge_d,h_edge_l:h_edge_r] = img
    # output sides
    out[:v_edge_u,h_edge_l:h_edge_r] = np.flipud(img[:v_edge_u,:]) # top
    out[v_edge_d:,h_edge_l:h_edge_r] = np.flipud(img[v_edge_d:,:]) # bottom
    out[v_edge_u:v_edge_d,:h_edge_l] = np.fliplr(img[:,:h_edge_l]) # left
    out[v_edge_u:v_edge_d,h_edge_r:] = np.fliplr(img[:,h_edge_r:]) # right
    # output corners
    out[:v_edge_u,:h_edge_l] = np.fliplr(out[:v_edge_u,h_edge_l:h_edge_l*2]) # top-left
    out[:v_edge_u,h_edge_r:] = np.fliplr(out[:v_edge_u,2*h_edge_r:h_edge_r]) # top-right
    out[v_edge_d:,:h_edge_l] = np.fliplr(out[v_edge_d:,h_edge_l:h_edge_l*2]) # bottom-left
    out[v_edge_d:,h_edge_r:] = np.fliplr(out[v_edge_d:,2*h_edge_r:h_edge_r]) # bottom-right
    return out

def augment_data(imgs, labels):
    '''
    A method to apply data augmentation on a list of imgs and their respective labels
    Input:
    :imgs: a list of images as uint16 numpy array
    :labels: a list of masks of the embilized areas of the images as boolean numpy array
    :out_size: a tuple of the desired output resolution
    Output:
    :imgs_aug: a list of augmented images as uint16 numpy array
    :labels_aug: a list of masks of the embilized areas of the augmented images as boolean numpy array
    '''
    imgs_aug, labels_aug = [],[]
    imgs_aug += imgs
    labels_aug += labels
    # add noisy versions
    n = len(imgs)
    noiseLvls = [0.2,0.1,0.05]
    for i in range(n):
        row,col = imgs[i].shape
        for noise in noiseLvls:
            imgs_aug.append(imgs[i]+np.random.normal(0.0,noise,(row,col))*65535)
            labels_aug.append(labels[i])
        imgs_aug.append(imgs[i]*(np.random.randn(row,col)*0.4+1))
        labels_aug.append(labels[i])

    # add rotated and flipped versions
    n = len(imgs_aug)
    rotations = [cv.ROTATE_90_CLOCKWISE, cv.ROTATE_180, cv.ROTATE_90_COUNTERCLOCKWISE]
    for i in range(n):
        img = imgs_aug[i]
        label = labels_aug[i]
        # add mirrored version
        imgs_aug.append(cv.flip(img,1))
        labels_aug.append(cv.flip(label,1))
        for r in rotations: # to cover all rotations
            angle = str((j+1)*90)
            # add rotated version 
            imgs_aug.append(cv.rotate(img,r))
            labels_aug.append(cv.rotate(label,r))
            # add rotated mirrored version
            imgs_aug.append(cv.rotate(cv.flip(img,1),r))
            labels_aug.append(cv.rotate(cv.flip(label,1),r))
    
    return imgs_aug, labels_aug


def mask_to_png(mask, color=[255,0,0]):
    '''convert nd array with 0 and 1 values to png image, transparent for 0, color of choice for 1
    
    input
    -----
    mask : nd array of 0 and 1 of uint8 (max 255)
    color : optionnal, array of red green blue colors of uint8 (max 255) values [r, g, b]
    
    output
    ------
    png image with transparent background and color at positions
    '''
    # NB opencv works with RGBA, so 4 dimensions, with A being alpha, the transparency (0=transparent)
    npmask = np.array(mask, dtype=np.uint8)
    npmaskr = npmask.copy()
    npmaskr[npmask!=0] = color[0] 
    npmaskg = npmask.copy()
    npmaskg[npmask!=0] = color[1]
    npmaskb = npmask.copy()
    npmaskb[npmask!=0] = color[2]
    npmaska = npmask.copy()
    npmaska[npmask!=0] = 255
    pngimage = np.stack([npmaskr,npmaskg, npmaskb, npmaska], axis=-1)
    
    return pngimage

def png_to_mask(png):
    '''convert nd array with 0 and 1 values to png image, transparent for 0, color of choice for 1
    
    input
    -----
    png : 4 or 3 (without alpha) dimensionnal nd array of 0 and color values (or transparency) of uint8 (max 255)
    
    output
    ------
    nd numpy array with 0 for background and 1 for color pixels
    '''
    # NB opencv works with RGBA, so 4 dimensions, with A being alpha, the transparency (0=transparent)
    pngarray = np.array(png)
#     print(pngarray[:,:,0])
    mask = pngarray[:,:,0]
    mask[pngarray[:,:,0]!=0]=1
    mask[pngarray[:,:,1]!=0]=1
    mask[pngarray[:,:,2]!=0]=1
    if pngarray.shape[2]==4:
        mask[pngarray[:,:,3]!=0]=1
        
    return mask

def segment_dataset(imgs, labels, in_size=572, out_size=388):
    '''
    A method to create a dataset ready to be used by a U-NET 
    Input:
    :imgs: a list of images as uint16 numpy array
    :labels: a list of masks of the embilized areas of the images as boolean numpy array
    :in_size: axis size of U-NET inputs
    :out_size: axis size of U-NET outputs
    Output:
    :X: a 3D numpy array of the inputs for the U-NET
    :y: a 3D numpy array of the outputs for the U-NET
    '''
    X, y = [], [] # lists of input and output data respectively
    ext = in_size - out_size # extand-mirror overall length
    for i, img in enumerate(imgs): # run through all images
        img_shp = np.array(img.shape) # store original image shape
        img_aug = extend_mirror(img, img_shp+ext) # extand-mirror input image
        segs = np.ceil(img_shp / out_size) # number of segments in each axis
        vg,hg = np.meshgrid(np.arange(segs[0]),np.arange(segs[1])) # create a grid of each axis
        grid = np.array([vg.ravel(),hg.ravel()]).T.astype(np.uint8) # create an array of segments coordinates
        for vh in grid: # run for each segment coordinate
            X_start = np.rint(vh*img_shp/segs + (ext/2 - in_size/2)*((0<vh)+(segs<=vh))).astype(np.uint16) # start pixel of input
            y_start = np.rint(vh*img_shp/segs - (out_size/2)*((0<vh)+(segs<=vh))).astype(np.uint16) # start pixel of output
            Xi = img_aug[X_start[0]:X_start[0]+in_size, X_start[1]:X_start[1]+in_size] # extract input segment
            yi = labels[i][y_start[0]:y_start[0]+out_size, y_start[1]:y_start[1]+out_size] # extract output segment
            X.append(Xi) # add to inputs list
            y.append(yi) # add to outputs list
    return np.array(X), np.array(y) # convert to np.array and return