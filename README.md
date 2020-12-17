# ML_FOOLS team Project 2 "ML for science" - Detecting the Degree of Cavitation In Situ in Young Trees

This project contains the work done for the Detecting the Degree of Cavitation In Situ in Young Trees project supervised by PERL as part of CS433-Machine Learning course at EPFL.
In this project, the air bubbles created in drought inside of young living trees were estimated and compared to the overall area of vessels existing in the tree. To do that the lab used X-ray microtomography on the same section of a living tree and later on the same section of the tree after cutting and flushing all the vessels.
Due to the differences between the shapes, amounts, locations, and the quality of the scans of the vessel between different trees, this is a tedious and not obvious manual task to do.

Therefore, the goal of the project from a machine learning side was to create a model that can label to a high accuracy the air bubbles on a slice image of a tree.
To do so the team developed a U-Net[[1]](#1) - a deep CNN with a contracting encoding and an expansive decoding paths that result in the likelihood for each class. 
The architecture, as seen in the image below, encodes the image by repeatingly using 2 consecutive 2D (3x3) Convolution layers followed by a ReLU activation and then a 2D (2x2) Max Pool. Then, the decoder similarly expands back the image by doing repeatedly 2 consecutive 2D (3x3) Convolution layers followed by an Upsample with factor 2 followed by a 2D (2x2) Convolution layer. The outputs of the decoders' 2D (2x2) Covoluional layers are concatenated with croped and copied data from the parallel level encoder. Finally, a 2D (1x1) Convolutional layer is used to set the output at the desired amount of labels.

![unet](https://user-images.githubusercontent.com/58084722/102468257-94ecaf00-4051-11eb-94b3-3d6b34b4474a.png)

## Repository structure
```
./
|---- run.py
|       > An executeable py file to run the model on data. More description below.
|
|---- helpers.py
|       > A python file that contains all the methods used for the different notebooks.
|
|---- generate_labels.ipynb
|       > A Jupyter notebook that generates the label images for the images.
|
|---- generate_model.ipynb
|       > A Jupyter notebook that generates the final model used.
|
|- other_models_nb/
|       > A folder that contains notebooks used to create the other notebooks discussed in the report.
|
|- csv_files
|       > A folder containing the CSV files used to select the images used for the labels.
|       |
|       |---- microCT_data_s1.csv
|               > A file with the analyzed data from all the good samples of Session 1
|       |
|       |---- microCT_data_s2.csv
|               > A file with the analyzed data from all the good samples of Session 2
|
|- models\
        |---- model.pickle
                > the final model pickle file
```

## Getting started
First ensure you have the tools and packages listed under. Then follow the steps to create the labels, model and finally final outputs.

### Requirements

The folowing packages and tools were used and needed to run the repository:\
[python](https://www.python.org/)==3.7 \
[ipython](https://ipython.org/notebook.html0==7.19.0 

[glob2](https://pypi.org/project/glob2/)==0.7 \
[pathlib2](https://pypi.org/project/pathlib2/)==2.3.5 

[opencv-python](https://opencv.org/)==4.4.0.46 \
[numpy](https://numpy.org/)==1.18.5 \
[scipy](https://www.scipy.org/)==1.5.4 \
[torch](https://pytorch.org/)==1.7.1 \
[matplotlib](https://matplotlib.org/)==3.3.2 \
[pandas](https://pandas.pydata.org/)==1.1.4 \
[seaborn](https://seaborn.pydata.org/)==0.11.0 

### Step 1: Generate label images

To generate the labels, open the notebook named "generate_labels.ipynb". In the notebook you will need to set the directories to which you want to store the labels to and select a CSV file containing the images you want to label with the analysis like ./.
The CSV file needs to contain 

### Step 2: Generate model


### Step 3: Predict vessels and generate analysis

### Jupyter notebooks

- xxx.ipynb
> descrip.

- th_how_to_use_png_masks.ipynb
> Small tutorial for easier use of masks in other places.

- th_ROI-EMB_toLabelLayers.ipynb
> Notebook going through the process of understanding the csv mask data provided, and converting it to png mask files

- th_matchFilePathToAnalysedPictures.ipynb
> Notebook in which we go through figuring out the provided images structure, and matching them to the data we have labels for.

- th_copyAllValidPicturesToLocal.ipynb
> Small file to copy all images from the Lab's drive to local folder for faster access.

## Data folder and files

- th_csv_labels (folder)
> Contains the csv provided mask files, information about them and the generated png masks/labels to be used in training

    - analysis_overview_df_with_pathnames.csv
   > Table containing one row per analysed plant, with the session, name, image number used, and voxel (pixel) size for the image.
    
    - csv_files_emb_roi.zip
   > Zipped folder containing the provided ROI and embolism selection in csv files.
   
    - png_masks_emb.zip
   > Zipped folder containing the generated png embolism masks/labels.

    - png_masks_roi.zip
   > Zipped folder containing the generated ROI embolism masks/labels.
   
- th_analysedimages (folder)
> Contains provided tif images that were analysed and for which we have label data. Images are re-named to the standardized Session[x]\_[flushed/living]\_[plant name]_[image number.tif_

- th_sampledata (folder)
> Contains some example tif and csv provided files. These are used in th_ROI-EMB_toLabelLayers.ipynb th_matchFilePathToAnalysedPictures.ipynb and th_copyAllValidPicturesToLocal.ipynb to get a hold of the whole dataset and implement the methods for all of them.


## References
<a id="1">[1]</a> O. Ronneberger, P. Fischer, & T. Brox, U-Net: Convolutional Networks for Biomedical Image Segmentation, arXiv, 2015
