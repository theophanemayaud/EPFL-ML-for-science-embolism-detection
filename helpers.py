import glob, os, ntpath
import numpy as np

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
    hist, bins = np.histogram(img.ravel(), np.arange(img.min(),img.max()+2,1))
    sigma_max = 0.0
    for t in range(1,len(hist)-1):
        w0 = np.nansum(hist[0:t])
        w1 = np.nansum(hist[t:-1])
        mu0 = np.nan_to_num(np.sum(hist[0:t]*bins[0:t]) / w0)
        mu1 = np.nan_to_num(np.sum(hist[t:-1]*bins[t:-2]) / w1)
        sigma = w0*w1*(mu0-mu1)**2
        if sigma > sigma_max:
            sigma_max = sigma
            t_otsu = t
    return t_otsu