# ML_FOOLS team Project 2 "ML for science"

## Files for final submission

In this folder you will find two main files :
- xx.py
>  descriptions ...

- xx.py
> descriptions.

## Files in which we tested different approaches

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

### Python files

- xxxx.py
> descript

- helpers.py
> Contains functions that are reused through multiple other files, like image pre-processing, conversions or other repetitive things.

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


