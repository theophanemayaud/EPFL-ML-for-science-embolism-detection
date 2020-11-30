import glob, os, ntpath
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label as region_map

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
    sigma_1 = np.std(img[0:range,0:range])
    sigma_2 = np.std(img[0:range,-range:-1])
    sigma_3 = np.std(img[-range:-1,-range:-1])
    sigma_4 = np.std(img[-range:-1,0:range])
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
    if np.any(img.shape>out_size):
        raise Exception('Error: at least on of out_size axes is smaller than the image shape')
    if np.any(3*img.shape>out_size):
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
