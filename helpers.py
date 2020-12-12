import glob, os, ntpath
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label as region_map
import cv2 as cv
import torch

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
    out = np.zeros(out_size) # initialize output image
    v_edge_u = (out_size[0]-img.shape[0]) // 2 # amount of pixels added to the top of the image
    v_edge_d = -(out_size[0]-img.shape[0]-v_edge_u) # amount of pixels added to the bottom of the image
    h_edge_l = (out_size[1]-img.shape[1]) // 2 # amount of pixels added to the left of the image
    h_edge_r = -(out_size[1]-img.shape[1]-h_edge_l) # amount of pixels added to the right of the image
    # output centre
    out[v_edge_u:v_edge_d,h_edge_l:h_edge_r] = img # copy the original to the center
    # output sides
    out[:v_edge_u,h_edge_l:h_edge_r] = np.flipud(img[:v_edge_u,:]) # extend mirror to the top
    out[v_edge_d:,h_edge_l:h_edge_r] = np.flipud(img[v_edge_d:,:]) # extend mirror to the bottom
    out[v_edge_u:v_edge_d,:h_edge_l] = np.fliplr(img[:,:h_edge_l]) # extend mirror to the left
    out[v_edge_u:v_edge_d,h_edge_r:] = np.fliplr(img[:,h_edge_r:]) # extend mirror to the right
    # output corners
    out[:v_edge_u,:h_edge_l] = np.fliplr(out[:v_edge_u,h_edge_l:h_edge_l*2]) # extend mirror to the top-left
    out[:v_edge_u,h_edge_r:] = np.fliplr(out[:v_edge_u,2*h_edge_r:h_edge_r]) # extend mirror to the top-right
    out[v_edge_d:,:h_edge_l] = np.fliplr(out[v_edge_d:,h_edge_l:h_edge_l*2]) # extend mirror to the bottom-left
    out[v_edge_d:,h_edge_r:] = np.fliplr(out[v_edge_d:,2*h_edge_r:h_edge_r]) # extend mirror to the bottom-right
    return out # return extended image

def augment_data(imgs, labels, noise=True, flip_rotate=False):
    '''
    A method to apply data augmentation on a list of imgs and their respective labels
    Input:
    :imgs: a list of images as uint16 numpy array
    :labels: a list of masks of the embilized areas of the images as boolean numpy array
    :noise: a boolean dictating if to add noisy versions of the samples
    :fliip_rotate: a boolean dictating if to add fliped and rotated versions of the samples
    Output:
    :imgs_aug: a list of augmented images as uint16 numpy array
    :labels_aug: a list of masks of the embilized areas of the augmented images as boolean numpy array
    '''
    imgs_aug, labels_aug = imgs, labels # lists of the augmented data
    # add noisy versions
    if noise:
        n = len(imgs) # amount of samples to augment
        noiseLvls = [0.2,0.1] # values of sigmas for gaussian distribution
        for i in range(n): # run for every sample
            row,col = imgs[i].shape # row and column sizes for random array sizes
            # add Gaussian noise
            for noise in noiseLvls:
                imgs_aug.append(imgs[i]+np.random.normal(0.0,noise,(row,col))*65535) # add noise to image
                labels_aug.append(labels[i]) # add matching label matrix
            # add speckle noise
            imgs_aug.append(imgs[i]*(np.random.randn(row,col)*0.4+1)) # add noise to image
            labels_aug.append(labels[i]) # add matching label matrix

    # add rotated and flipped versions
    if flip_rotate:
        n = len(imgs_aug) # amount of samples to augment
        rotations = [cv.ROTATE_90_CLOCKWISE, cv.ROTATE_180, cv.ROTATE_90_COUNTERCLOCKWISE] # values of rotations for cv.rotate()
        for i in range(n): # run for every sample
            img = imgs_aug[i] # current original image
            label = labels_aug[i] # current original label
            # add mirrored version
            imgs_aug.append(cv.flip(img,1)) # mirror image
            labels_aug.append(cv.flip(label,1)) # mirror label
            for r in rotations: # to cover all rotations
                # add rotated version 
                imgs_aug.append(cv.rotate(img,r)) # rotate image
                labels_aug.append(cv.rotate(label,r)) # rotate label
                # add rotated mirrored version
                imgs_aug.append(cv.rotate(cv.flip(img,1),r)) # flip and rotate image
                labels_aug.append(cv.rotate(cv.flip(label,1),r)) # flip and rotate label
        
    return imgs_aug, labels_aug # return augmented samples


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

def compute_emb_surf_pred_error(original_label, predicted_label, print_values=False):
    '''From the original and predicted labels compute surface % error(0% is perfect)
    
    Note : 0% is a perfect score. 
            Positive values means there are more predicted pixels than actual, 
            ex 100% means 2x more predicted pixels, 
            200% means 3x as many predicted pixels. 
            -100% means 2x more original pixels,  
            -200% means 3x more original pixels
            This is 
    
    input
    -----
    original_label : 2d numpy array of 1 and 0 only
    predicted_label : 2d numpy array of 1 and 0 only of same size as original_label
    print_values : bolean whether the function should print number of pixels
                    at 1 for each
    
    output
    ------
    error of surface prediction in percentage (can be positive and negative, 
        if predicted more or less than should have). The ideal is 
        0% : no over or under pixels were predicted.
    '''
    if original_label.shape != predicted_label.shape:
        raise NameError("Inputs must be 2d numpy arrays of same sizes")
    ori_lab_counts = np.count_nonzero(original_label == 1)
    pred_lab_counts = np.count_nonzero(predicted_label == 1)
    
    if print_values==True:
        print(f"Original label 1s={ori_lab_counts}, Predicted label 1s={pred_lab_counts}")

    if ori_lab_counts == 0:
        ori_lab_counts = 1; #fix when some masks are 0 to not have divide by 0
    if pred_lab_counts == 0:
            pred_lab_counts = 1; #fix when some predictions are 0 to not have divide by 0
    if pred_lab_counts >= ori_lab_counts:
        emb_surf_pred_error = 100*(pred_lab_counts-ori_lab_counts)/ori_lab_counts
    else:
        emb_surf_pred_error = -100*(ori_lab_counts-pred_lab_counts)/pred_lab_counts
    return emb_surf_pred_error

def confusion(pred, test_labels, data_type):
    '''Calculate the percentage of true-positive, true-negative, false-positive, and false-negative
    
    Input :
    pred: prediction of the labels, either tensor or numpy array
        if torch.tensor (result of U-net) of size [num_batches=1, 2, dim_image1, dim_image2],
        if numpy must be 2d array of 1 and 0
    test_labels: Real labels for the image, either tensor or numpy array
        if torch.tensor () must be same dimensions as prediction
        if numpy, must be 2d array of 1 and 0 of same dimensions as prediction
    data_type: string either tensor or numpy, indicating type on pred and test_labels
    
    Return:
    TP, FP, TN, TP rates as ratio over 1
    '''
    if data_type == "torch":
        pred_, lbls_ = torch.argmax(pred,dim=1).view(-1), test_labels.view(-1)
        TP = torch.sum(torch.logical_and(pred_==1, lbls_==1)) / torch.sum(lbls_==1)
        TN = torch.sum(torch.logical_and(pred_==0, lbls_==0)) / torch.sum(lbls_==0)
        FP = torch.sum(torch.logical_and(pred_==1, lbls_==0)) / torch.sum(lbls_==0)
        FN = torch.sum(torch.logical_and(pred_==0, lbls_==1)) / torch.sum(lbls_==1)
    elif data_type == "numpy":
        TP = ((pred==1)*(test_labels==1)).sum() / (test_labels==1).sum()
        TN = ((pred==0)*(test_labels==0)).sum() / (test_labels==0).sum()
        FP = ((pred==1)*(test_labels==0)).sum() / (test_labels==0).sum()
        FN = ((pred==0)*(test_labels==1)).sum() / (test_labels==1).sum()
    else:
        raise NameError("You must specify a type !!!")

    return TP, FP, TN, TP

def segment_dataset(imgs_, labels_, in_size=572, out_size=388, extend = True, augment=[False, False]):
    '''
    A method to create a dataset ready to be used by a U-NET 
    Input:
    :imgs: a list of images as uint16 numpy array
    :labels: a list of masks of the embilized areas of the images as boolean numpy array
    :in_size: axis size of U-NET inputs
    :out_size: axis size of U-NET outputs
    :extend: boolean dictating if to apply image mirror-extension or not
    :augment: boolean dictating if to apply data augmentation or not
    Output:
    :X: a 3D numpy array of the inputs for the U-NET
    :y: a 3D numpy array of the outputs for the U-NET
    '''
    X, y = [], [] # lists of input and output data respectively
    if extend:
        ext = in_size - out_size # extand-mirror overall length
    else:
        ext = 0 # extention size is 0 in the case of no extention
    if np.any(augment):
        imgs, labels = augment_data(imgs_, labels_, noise=augment[0], flip_rotate=augment[1]) # augmented images and labels lists
    else:
        imgs, labels = imgs_, labels_ # maintain original images and labels in the case of no augmentation
    for i, img in enumerate(imgs): # run through all images
        img_shp = np.array(img.shape) # store original image shape
        if extend:
            img_aug = extend_mirror(img, img_shp+ext) # extand-mirror input image
        else:
            img_aug = img # original input in the case of no extention
        segs = np.ceil(img_shp / out_size) # number of segments in each axis
        vg,hg = np.meshgrid(np.arange(segs[0]),np.arange(segs[1])) # create a grid of each axis
        grid = np.array([vg.ravel(),hg.ravel()]).T.astype(np.uint8) # create an array of segments coordinates
        ol_block = (img_shp - out_size) / (segs - 1) # calculate size of overlapping blocks
        for vh in grid: # run for each segment coordinate
            start = np.rint(ol_block*vh).astype(np.uint16) # calculate start pixel 
            Xi = img_aug[start[0]:start[0]+in_size, start[1]:start[1]+in_size] # slice input segment
            yi = labels[i][start[0]:start[0]+out_size, start[1]:start[1]+out_size] # slice output segment
            X.append(Xi) # add to inputs list
            y.append(yi) # add to outputs list
    return np.array(X), np.array(y) # convert to np.array and return

def out_predict(model, img, device, in_size=572, out_size=388, extend=True):
    '''
    A method to create a dataset ready to be used by a U-NET 
    Input:
    :model: the trained U-NET model
    :img: 2D numpy array with the image
    :device: CUDA or CPU to use for the tensors
    :in_size: axis size of U-NET inputs
    :out_size: axis size of U-NET outputs
    :extend: boolean dictating if to apply image mirror-extension or not
    Output:
    :pred: 2D numpy array with the prediction
    '''
    if extend:
        ext = in_size - out_size # extand-mirror overall length
    else:
        ext = 0
    img_shp = np.array(img.shape) # store original image shape
    if extend:
        img_ext = extend_mirror(img, img_shp+ext) # extand-mirror input image
    else:
        img_ext = img # original input in the case no extention
    segs = np.ceil(img_shp / out_size) # number of segments in each axis
    vg,hg = np.meshgrid(np.arange(segs[0]),np.arange(segs[1])) # create a grid of each axis
    grid = np.array([vg.ravel(),hg.ravel()]).T.astype(np.uint8) # create an array of segments coordinates
    # segs -= 1 # change segs to axis index value limits
    ol_block = (img_shp - out_size) / (segs - 1) # calculate size of overlapping blocks
    pred_max_logit = np.ones((2,img_shp[0],img_shp[1])) * np.nan # max probability 
    pred = np.zeros_like(img)
    for vh in grid: # run for each segment coordinate
        start = np.rint(ol_block*vh).astype(np.uint16) # calculate start pixel
        Xi = img_ext[start[0]:start[0]+in_size, start[1]:start[1]+in_size] # slice input segment
        # Create input tensor
        tensor_X = torch.Tensor(Xi).view(1,1,in_size,in_size) # create a Tensor of segment input
        tensor_X = tensor_X.to(device) # send to GPU/CPU
        # Predict output and select best overlapping prediction
        prediction = model(tensor_X).detach() # calculate prediction
        prediction = prediction.cpu().numpy().squeeze() # convert to numpy matrix
        pred_max_logit[:,start[0]:start[0]+out_size, start[1]:start[1]+out_size] = np.fmax(prediction,pred_max_logit[:,start[0]:start[0]+out_size, start[1]:start[1]+out_size]) # check for class max lieklihood for each overlapping pixel
        pred[start[0]:start[0]+out_size, start[1]:start[1]+out_size] = np.argmax(pred_max_logit[:,start[0]:start[0]+out_size, start[1]:start[1]+out_size],axis=0) # determine class of segment's pixels
    return pred # convert to np.array and return
