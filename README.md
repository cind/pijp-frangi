# pijp-frangi

This pipeline uses the Frangi filter ( {Multiscale Vessel Enhancement Filtering} ) and various preprocessing steps to segment perivascular spaces.
Creator: Serena Tang

## Current processing pipeline (for FS-processed images):
Stage:
- convert the T1 images to nii
- convert aseg.stats to csv
- make a white matter mask
    - segmentations are picked from wmparc from FS, only the white matter portions that are counted in Sepehrband et al 2021
    - mask is put together then closed (radius 1, cross shape)
- make a grey matter mask
    - segmentations are taken from aparc+aseg.mgz, put together like white matter mask (but not closed)
- option for WMH mask, currently using v.1 (v.2 has issues with mask union sometimes / sampling ? 2/2/24)
    - v.1: register FLAIR to T1 and bias correct (Sepehrband et al 2021)
    - v.1: LST algorithm finds all the WMH
    - v.1: Mask is then dilated by 1 with cross shape
    - v.2: Threshold FLAIR image to 70% and add this thresholded mask to the WMH mask produced in v.1
    - (latest recommendation is to use just v.1. If v.2, have to run both functions (v.1, v.2))

Analyze:
- run frangi filter with default recommended parameters (threshold at 0.0002 (.0004 for RAW), close to Sepehrband et al 2021), removing WMH with WMH mask if it exists ***change from 0.0002 to 0.00002 and 0.00004 2/26/24***
- remove blobs that are likely too big to be PVS (needs to be checked) ***added 1/4/24***
- remove anything that is 3 voxels (noise) ***added 2/26/24***
- calculate with aseg.stats
- run frangi filter with just white matter mask and default parameters (threshold is the same as above), removing WMH with WMH mask if it exists
- calculate the mask components (using connected components analysis) then measure how many components there are and how large the components are (gets volume and count)
- put all PVS stats into a csv for each subject, stored in subject folder
- put subject info into grand report 


## Current processing pipeline (for RAW images):
- Main difference is only bias field correction is run + denoise (p=1, r=2) is run, nothing else
- Everything else is the same (uses the same masks)









