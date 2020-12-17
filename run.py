# import libraries
import glob, os, ntpath, sys, pathlib
import numpy as np
import pandas as pd
import cv2 as cv
import torch
import torch.nn as nn
import xml.etree.ElementTree as ET
from scipy.ndimage import label as region_map

def main():
    # create a dictionary for the CSV 
    csv_dict = {'plant_name':[], 'slice':[], 'vessle_num':[], 'surface':[], 'diameter':[], 'x-coordinate':[], 'y-coordinate':[]}

    # list all XML files to get plant names and voxel sizes
    xml_paths_l = sorted(glob.glob(path_l+'**\\*.xml'))
    xml_paths_f = sorted(glob.glob(path_f+'**\\*.xml'))

    # set device
    if not torch.cuda.is_available():
        print("Warning: Things will go much quicker if you enable a GPU or use a device with access to a GPU")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load models and send to GPU
    model = torch.load(models_path+'model.pkl')
    model.to(device)

    # run prediction and analysis for all living plants images in subdirectories
    for xml_path in xml_paths_l:
        voxel_size = extract_voxel(xml_path) # extract the voxel size of the scans
        plant_name = os.path.basename(os.path.dirname(xml_path)) # remove file name then keep only the directory name
        path = pathlib.Path('.\\output\\living\\'+plant_name) # set plant output directory
        path.mkdir(parents=True, exist_ok=True) # creates folder if doesn't exist
        for img_path in np.array(sorted(glob.glob(path_l+f'**{plant_name}\\*'+img_type)),dtype=object): # go through all sacns in folder
            slice_name = extract_name(img_path) # extract slice name
            img = cv.imread(img_path, cv.IMREAD_UNCHANGED)/(0xFFFF) # load image
            pred = out_predict(model, img, device) # predict embolized area

            # store preicted image overlayed on top of original image
            pred_ol = np.repeat(img[:, :, np.newaxis], 3, axis=2) # convert image from grayscale to RGB
            pred_ol[pred==1] = np.array([0xFFFF,0,0]) # overlay prediction over original image
            cv.imwrite(f'.\\output\\living\\{plant_name}\\{slice_name}{img_type}', pred_ol) # store overlayed image

            # analyze embolized areas
            vessels_map, vessels = region_map(pred) # create a regions map for the embolized areas
            for vessel in range(vessels): # run for each area
                vessel_pixels = np.argwhere(vessels_map==vessel+1) # an array of what pixels contain the vessel
                surface = voxel_size**2 * len(vessel_pixels) # calculate the surface area of the vessel in mm^2 
                diameter = np.sqrt(4 * surface / np.pi) # calculate the equivalent circle diameter in mm
                yx = voxel_size * (np.max(vessel_pixels, axis=0) + np.min(vessel_pixels, axis=0)) / 2 # calculate equivalent circle centre coordinate in mm

                # store all data into dictionary
                csv_dict['plant_name'].append(plant_name)
                csv_dict['slice'].append(slice_name)
                csv_dict['vessle_num'].append(vessel)
                csv_dict['surface'].append(surface)
                csv_dict['diameter'].append(diameter)
                csv_dict['x-coordinate'].append(yx[1])
                csv_dict['y-coordinate'].append(yx[0])

    # run prediction and analysis for all flushed plants images in subdirectories
    for xml_path in xml_paths_f:
        voxel_size = extract_voxel(xml_path) # extract the voxel size of the scans
        plant_name = os.path.basename(os.path.dirname(xml_path)) # remove file name then keep only the directory name
        path = pathlib.Path('.\\output\\flushed\\'+plant_name) # set plant output directory
        path.mkdir(parents=True, exist_ok=True) # creates folder if doesn't exist
        for img_path in np.array(sorted(glob.glob(path_f+f'**{plant_name}\\*'+img_type)),dtype=object): # go through all sacns in folder
            slice_name = extract_name(img_path) # extract slice name
            img = cv.imread(img_path, cv.IMREAD_UNCHANGED)/(0xFFFF) # load image
            pred = out_predict(model, img, device) # predict embolized area

            # store preicted image overlayed on top of original image
            pred_ol = np.repeat(img[:, :, np.newaxis], 3, axis=2) # convert image from grayscale to RGB
            pred_ol[pred==1] = np.array([0xFFFF,0,0]) # overlay prediction over original image
            cv.imwrite(f'.\\output\\flushed\\{plant_name}\\{slice_name}{img_type}', pred_ol) # store overlayed image

            # analyze embolized areas
            vessels_map, vessels = region_map(pred) # create a regions map for the embolized areas
            for vessel in range(vessels): # run for each area
                vessel_pixels = np.argwhere(vessels_map==vessel+1) # an array of what pixels contain the vessel
                surface = voxel_size**2 * len(vessel_pixels) # calculate the surface area of the vessel in mm^2 
                diameter = np.sqrt(4 * surface / np.pi) # calculate the equivalent circle diameter in mm
                yx = voxel_size * (np.max(vessel_pixels, axis=0) + np.min(vessel_pixels, axis=0)) / 2 # calculate equivalent circle centre coordinate in mm

                # store all data into dictionary
                csv_dict['plant_name'].append(plant_name+'_dry')
                csv_dict['slice'].append(slice_name)
                csv_dict['vessle_num'].append(vessel)
                csv_dict['surface'].append(surface)
                csv_dict['diameter'].append(diameter)
                csv_dict['x-coordinate'].append(yx[1])
                csv_dict['y-coordinate'].append(yx[0])

    # create CSV from dictionary
    pd.DataFrame.from_dict(csv_dict).to_csv('.\\output\\analyzed_data.csv', index=False)
    print(f'Process complete!!\nThe overlayed living images are in {'.\\output\\living\\<plant_name>'}.\nThe overlayed flushed images are in {'.\\output\\flushed\\<plant_name>'}.\nThe CVS file  is in {.\\output\\analyzed_data.csv})


#############################
# Command line Help methods #
#############################

def help():
    help_s = '''
*****************************************

usage: pipeline.py living_plants_path flushed_plants_path [options] ...

*****************************************

This is a tool to detect and analyze the ratio between embolism regions in living and flushed plants. The tool requires seperate paths to the top directories containing all the living and flushed plant images.

living_plants_path      A path to the top directory of all living plants images.
flushed_plants_path     A path to the top directory of all flushed plants images.
**Note**: By default the tool will analyze all .TIF files in the subdirectories from the paths, but the type could be modified by [-type <type>]

*****************************************

Options:
    -h, -help, --h, --help  Displays this page. Shows instructions on how to use the tool and its different options.
    -model <models_path>    Modifies the location of the models. By defult models folder is expected at .\models. 
                            The names of the model files need to be model._l.pickle and model_f.pickle for the living plants and flushed plants respectively.
    -type <img_type>        Modifies type of the images used. By defult the type is TIF.

*****************************************

Example:
    pipeline.py .\\living\\ .\\flushed\\ -model .\\models\\ -type PNG


*****************************************
        '''
    print(help_s)

###################
# Support methods #
###################

# A metod the extrac the voxel value of the scans
def extract_voxel(xml_path):
    '''
    A method to extrac the voxel value of the scans

    Input:
    :xml_path: a string of the path name of the XML file from the scan
    
    Output:
    :voxel_size: float of the voxel size
    '''
    # extract voxel size from XML file
    tree = ET.parse(xml_path) # get XML tree from file
    root = tree.getroot() # get the root of the XML tree
    voxel_size = root[0][1][0].attrib['X'] # get <conebeam\volume_acquisition\voxelSize['X']>

    return float(voxel_size)


# A method to exrtact just the file name from a path
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


# A method that extend and mirror the edges on an image to prepare it to the model
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

# A method to predict the output of an image through a model
def out_predict(model, img, device, in_size=572, out_size=388, extend=True):
    '''
    A method that segments an image to overlapping segments in the model input shape, predicts each segment, and construct the full image prediction
    
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
    model.eval() # set model for prediction
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


################################################
# U_NET pytorch architecture and subcomponents #
################################################

# UNet definitions
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        # functions for going down the U
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.d_double_conv_1 = double_conv(1, 64)
        self.d_double_conv_2 = double_conv(64, 128)
        self.d_double_conv_3 = double_conv(128, 256)
        self.d_double_conv_4 = double_conv(256, 512)
        self.d_double_conv_5 = double_conv(512, 1024)
        
        # functions for going up the U
        self.up_trans_4 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)        
        self.u_double_conv_4 = double_conv(1024, 512)
        self.up_trans_3 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.u_double_conv_3 = double_conv(512, 256)
        self.up_trans_2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.u_double_conv_2 = double_conv(256, 128)
        self.up_trans_1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.u_double_conv_1 = double_conv(128, 64)
        
        self.out = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1)
        
    def forward(self, image):
        '''makes the 388x388 prediction with the model, image must be 572x572pixels'''
        
        # Going down the U
        d1 = self.d_double_conv_1(image) # first "level"
        # print(x1.size())
        x = self.max_pool_2x2(d1)
        d2 = self.d_double_conv_2(x) # second
        x = self.max_pool_2x2(d2)
        d3 = self.d_double_conv_3(x) # third
        x = self.max_pool_2x2(d3)
        d4 = self.d_double_conv_4(x) # fourth
        x = self.max_pool_2x2(d4)
        x = self.d_double_conv_5(x) # last layer (fifth) : no max pool
        
        # Going up the U
        x = self.up_trans_4(x)
        d4 = crop_img(tensor=d4, target_tensor=x) #crop to copy
        x = self.u_double_conv_4(torch.cat([d4, x], 1))
        
        x = self.up_trans_3(x)
        d3 = crop_img(tensor=d3, target_tensor=x)
        x = self.u_double_conv_3(torch.cat([d3, x], 1))
        
        x = self.up_trans_2(x)
        d2 = crop_img(tensor=d2, target_tensor=x)
        x = self.u_double_conv_2(torch.cat([d2, x], 1))
        
        x = self.up_trans_1(x)
        d1 = crop_img(tensor=d1, target_tensor=x)
        x = self.u_double_conv_1(torch.cat([d1, x], 1))
        
        x = self.out(x)
        return x
    
# some functions so reduce redunduncy
def double_conv(nb_in_channels, nb_out_channels): # Used for every descending step
    conv = nn.Sequential(
        nn.Conv2d(nb_in_channels, nb_out_channels, kernel_size=3),
        nn.ReLU(inplace=True),
        nn.Conv2d(nb_out_channels, nb_out_channels, kernel_size=3),
        nn.ReLU(inplace=True),
    )
    return conv

def crop_img(tensor, target_tensor): # Used for copy and crop between descending and ascending
    target_size = target_tensor.size()[2] # NB they are square so .size[2]=.size[3]
    tensor_size = tensor.size()[2]
    delta = tensor_size - target_size #target is always smaller
    pix_crop = delta // 2
    return tensor[:, :, pix_crop:tensor_size-pix_crop, pix_crop:tensor_size-pix_crop]

########################
# Command line handler #
########################

if __name__=='__main__':
# command line options
    # Handling models path
    if np.any(sys.argv == '-model'): # check if model option used
        models_path = sys.argv[np.argmax(np.array(sys.argv,dtype=np.object)=='-model')+1]
    else: # otherwise use defult path
        models_path = './models/'
    # Handling img type
    if np.any(sys.argv == '-type'): # check if type option used
        img_type = '.'+sys.argv[np.argmax(np.array(sys.argv,dtype=np.object)=='-type')+1] 
    else: # otherwise use defult type
        img_type = '.tif'
    # Handling help or too little parameters
    if np.any(sys.argv == ['-h','-help','--h','--help']) or len(sys.argv)<3:
        help()
    else: # otherwise run main
        path_l = sys.argv[1]
        path_f = sys.argv[2]
        main()