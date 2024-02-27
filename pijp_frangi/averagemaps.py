import os
import subprocess
import logging
import argparse
import textwrap
import string
import datetime

import numpy as np
import shutil
import pandas as pd
import sys
import gzip
import nibabel as nib
from matplotlib import pyplot as plt
from scipy import stats
import glob

parent = '/m/InProcess/External/ADNI3/ADNI3_frangi/pijp-frangi'
rg = 'AD'
paths = os.listdir(parent,rg)
subjects = [s for s in paths if not(s.startswith('.'))]

stats = []
for s in subjects:
    img = nib.load(os.path.join(parent,rg,s,s+'-frangi-thresholded-wmhrem.nii.gz'))
    data = img.get_fdata()
    stats.append(data)

stats = np.array(stats)
avg = np.sum(stats,axis=0) / stats.shape[0]

maskname = os.path.join(parent,rg,'average'+rg+'map.nii.gz')
maskimg = nib.Nifti1Image(avg, img.affine)
nib.save(maskimg, maskname)