import os
import numpy as np
import shutil
import pandas as pd
import sys
import gzip
import nibabel as nib
from matplotlib import pyplot as plt
from scipy import stats


path_subj = '/Users/nanatang/Documents/GradResearch/ADNI3_samples_fsseg'
files = os.listdir(path_subj)
subjects = [i for i in files if i.startswith('sub')]
toy = os.path.join(path_subj,subjects[0],'aparc+aseg.mgz')

img = nib.load(toy)
data = img.get_fdata()
mask = np.zeros(np.shape(data))

seg = [2,10,11,12,13,26,41,49,50,51,52,58]
wmh = 77    # do we need this if it should already not be included in the others? or is it included? or maybe it doesn't matter since 
            # she wants me to use SPM anyway

for m in seg:
    mask[data == m] = m

maskimg = nib.Nifti1Image(mask,img.affine)
nib.save(maskimg,os.path.join(path_subj,subjects[0],'test-mask.nii'))