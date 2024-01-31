# pijp-frangi

This pipeline recreates a process from the paper, {Multiscale Vessel Enhancement Filtering}.

Current processing pipeline:

Stage:
- convert the T1 images to nii
- convert aseg.stats to csv
- make a white matter mask
    - segmentations are picked from wmparc from FS, only the white matter portions that are counted in Sepehrband et al 2021
    - mask is put together then closed (radius 1, cross shape)
- make a grey matter mask
    - segmentations are taken from aparc+aseg.mgz, put together like white matter mask (but not closed)
- option for WMH mask, currently using v.2
    - v.1: register FLAIR to T1 and bias correct (Sepehrband et al 2021)
    - v.1: LST algorithm finds all the WMH
    - v.1: Mask is then dilated by 1 with cross shape
    - v.2: Threshold FLAIR image to 70% and add this thresholded mask to the WMH mask produced in v.1
    - (latest recommendation is to use just v.1. If v.2, have to run both functions (v.1, v.2))

Analyze:
- run frangi filter with default recommended parameters (threshold at 0.0002, close to Sepehrband et al 2021), removing WMH with WMH mask if it exists
- remove blobs that are likely too big to be PVS (needs to be checked) ***added 1/4/23***
- calculate with aseg.stats
- run frangi filter with just white matter mask and default parameters (threshold is the same as above), removing WMH with WMH mask if it exists
- calculate the mask components (using connected components analysis) then measure how many components there are and how large the components are (gets volume and count)
- put everything into excel sheet 







