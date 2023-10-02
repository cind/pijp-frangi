import os
import numpy as np
import shutil
import pandas as pd
import sys
import gzip
import nibabel as nib
from matplotlib import pyplot as plt
from scipy import stats

import frangi
import repo
from pijp.core import Step, get_project_dir


# loop through all the subjects
# call each method in frangi.py
# make a table for each subject for pvs count/volume

#** just trying this in freesurfer for now


project = 'ADNI3_frangi'

# idk how to get a list of subjects
data_dir = '/m/InProcess/External/ADNI3_FSdn/Freesurfer/subjects/'
subjects = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir,d)) & d.startswith('ADNI')]


subject_codes = []
researchgroup = []
pvscount = []
pvsvol = []
icvnorm_vol = []


for code in subjects:

    basics = frangi.BaseStep(project, code)
    prep = frangi.Stage(project, code)
    analyze = frangi.Analyze()

    t1mgz = os.path.join(basics.mrifolder,'T1.mgz')
    maskmgz = os.path.join(basics.mrifolder,'aparc+aseg.mgz')
    asegstats = os.path.join(basics.statsfolder,'aseg.stats')

    #prep.mgz_convert(t1mgz,idk??)
    prep.aseg_convert(asegstats)
    prep.make_mask(maskmgz)

    analyze.frangi_analysis(t1nii, basics.wmdgmask, 0.0025)
    analyze.icv_calc(basics.asegstats)
    analyze.pvs_stats(basics.frangimask)

    subject_codes.append(code)
    researchgroup.append(basics.researchgroup)
    pvscount.append(analyze.count)
    pvsvol.append(analyze.vol)
    icvnorm_vol.append(analyze.icv_normed)

basics = frangi.BaseStep(project, code)



col = ['subjects','research group','pvsvol','pvscount','icv norm']

alldata = pd.DataFrame(data=zip(subject_codes,researchgroup,pvsvol,pvscount,icvnorm_vol),index=np.arange(len(subject_codes))+1,columns=col)

alldata.to_csv(os.path.join(get_project_dir(basics.project),'frangidata.csv'), index=True)