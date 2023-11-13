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
import glob

import frangi
import repo
from pijp.core import Step, get_project_dir

#** just trying this in freesurfer for now


project = 'ADNI3_frangi'
data_dir = '/m/InProcess/External/ADNI3_FSdn/Freesurfer/subjects/'
subjects = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir,d)) & d.startswith('ADNI')][0:1]
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
    
    flairtest = glob.glob('/m/InProcess/External/ADNI3/ADNI3_frangi/Raw/'+code[0:19]+'*.FLAIR.nii.gz')
    rg = repo.Repository(project).get_researchgroup(code)
    #basics = frangi.BaseStep(project, code)
    if len(rg) > 0 & (len(flairtest) > 0):
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
        basics.make_allmask(maskmgz)


        # get the flair file, make wmh mask
        flairraw = glob.glob(get_project_dir(basics.project)+'/Raw/'+code[0:19]+'/*.FLAIR.nii.gz')[0]
        print(flairraw)
        #flairraw = os.path.join(basics.project,'Raw',code[0:11],code+'.FLAIR.nii.gz')
        basics.make_wmhmask(basics.t1,flairraw)

        sys.exit()


        frangimask_all = os.path.join(basics.working_dir, basics.code + "-frangi-thresholded-wmhrem.nii.gz")
        analyze.frangi_analysis(basics.t1, basics.allmask, 0.0025, frangimask_all,basics.wmhmask)

        analyze.icv_calc(basics.asegstats)
        analyze.pvs_stats(frangimask_all)
        pvscount.append(analyze.count)
        pvsvol.append(analyze.vol)
        icvnorm_vol.append(analyze.icv_normed)


        frangimask_wm = os.path.join(basics.working_dir, basics.code + "-frangi-thresholded-wm-wmhrem.nii.gz")
        analyze.frangi_analysis(basics.t1, basics.wmmask, 0.0002, frangimask_wm,region = 'wm',wmhmask = basics.wmhmask)

        analyze.pvs_stats(frangimask_wm)
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