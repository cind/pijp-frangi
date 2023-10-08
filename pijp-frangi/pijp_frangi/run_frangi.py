import os
import numpy as np
import shutil
import pandas as pd
import sys
import gzip
import nibabel as nib
from matplotlib import pyplot as plt
from scipy import stats
import logging

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
subjects = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir,d)) & d.startswith('ADNI')][0:100]
print(subjects)

subject_codes = []
researchgroup = []
pvscount = []
pvsvol = []
icvnorm_vol = []

pvscountwm = []
pvsvolwm = []
icvnorm_volwm = []


for code in subjects:
    
    rg = repo.Repository(project).get_researchgroup(code)
    #basics = frangi.BaseStep(project, code)
    if len(rg) > 0:
        basics = frangi.Stage(project, code)
        analyze = frangi.Analyze(project, code)

        # getting the mgz files
        t1mgz = os.path.join(basics.mrifolder,'T1.mgz')
        wmparcmgz = os.path.join(basics.mrifolder,'wmparc.mgz')
        maskmgz = os.path.join(basics.mrifolder,'aparc+aseg.mgz')
        asegstats = os.path.join(basics.statsfolder,'aseg.stats')

        # converting mgz and making masks
        basics.mgz_convert(t1mgz,basics.t1)
        basics.mgz_convert(wmparcmgz,basics.wmmask)
        basics.aseg_convert(asegstats)
        basics.make_mask(maskmgz)

        analyze.icv_calc(basics.asegstats)

        analyze.frangi_analysis(basics.t1, basics.allmask, 0.0025, basics.frangimask_all)
        analyze.pvs_stats(basics.frangimask_all)
        pvscount.append(analyze.count)
        pvsvol.append(analyze.vol)
        icvnorm_vol.append(analyze.icv_normed)

        analyze.frangi_analysis(basics.t1, basics.wmmask, 0.0002, basics.frangimask_wm,region='wm')
        analyze.pvs_stats(basics.frangimask_wm)
        pvscountwm.append(analyze.count)
        pvsvolwm.append(analyze.vol)
        icvnorm_volwm.append(analyze.icv_normed)

        subject_codes.append(code)
        researchgroup.append(basics.researchgroup)

    

    else:
        print('This code doesnt work: ',code)

        

#basics = frangi.BaseStep(project, code)



col = ['subjects','research group','pvsvol','pvscount','icv norm','pvsvolwm','pvscountwm','icv norm wm']
alldata = pd.DataFrame(data=zip(subject_codes,researchgroup,pvsvol,pvscount,icvnorm_vol,pvsvolwm,pvscountwm,icvnorm_volwm),index=np.arange(len(subject_codes))+1,columns=col)
alldata.to_csv(os.path.join(get_project_dir(basics.project),'frangidata.csv'), index=True)