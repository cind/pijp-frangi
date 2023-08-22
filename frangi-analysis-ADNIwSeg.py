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
#print(subjects)
t1_list = []
#flair_list = []
#gt_list = []



# create a new folder in analysis folder for each subject, and gather all T1s into a list and all ground truths into a list
for subject in subjects:
    # subjdir = os.path.join(os.getcwd(),'frangi-valdo',subject+'-frangi')    # /Users/nanatang/Documents/GradResearch/frangi-valdo/sub-101-frangi
    # os.makedirs(subjdir,exist_ok=True)
    t1 = os.path.join(path_subj,subject,subject+'-T1.nii.gz')  # '/Users/nanatang/Documents/GradResearch/SYT-MRIimages/Task1-selectfullseg/sub-101/sub-101-..T1.nii.ga'

    # maybe don't need to rotate since you won't be doing any preprocessing

    # # rotate the image bc somehow that works ??
    # img = nib.load(t1)
    # data = img.get_fdata()
    # data_rot = np.rot90(data,axes=(2,1))
    # img_rot = nib.Nifti1Image(data_rot,np.eye(4))
    # nib.save(img_rot,os.path.join(path_subj,subject,subject+'-T1rot.nii.gz'))  
    # t1_rot = os.path.join(path_subj,subject,subject+'-T1rot.nii.gz')

    #shutil.copy(t1,subjdir)
    t1_list.append(t1)
 
### commands ###
#qit command
qit = '/Users/nanatang/anaconda3/envs/pvssegment/bin/qit-build-mac-latest/bin/qit'

#ANTs commands
bias_command = '/Users/nanatang/anaconda3/envs/pvssegment/build/ANTS-build/Examples/N4BiasFieldCorrection'
denoise_command = '/Users/nanatang/anaconda3/envs/pvssegment/build/ANTS-build/Examples/DenoiseImage'

#FSL commands
fsl_segment = '/Users/nanatang/Documents/GradResearch/fsl/bin/fast'
fsl_brain = '/Users/nanatang/Documents/GradResearch/fsl/bin/bet'
fsl_deepgrey = '/Users/nanatang/Documents/GradResearch/fsl/bin/run_first_all'
fsl_firstflirt = '/Users/nanatang/Documents/GradResearch/fsl/bin/first_flirt'
fsl_runfirst = '/Users/nanatang/Documents/GradResearch/fsl/bin/run_first'
fsl_maths = '/Users/nanatang/Documents/GradResearch/fsl/bin/fslmaths'

# #LST commands, addpath
# matlab = '/Applications/MATLAB_R2023a.app/bin/matlab'
# addpath = "\"addpath('/Users/nanatang/Documents/GradResearch/spm12');exit\""
# os.system(f'{matlab} \
#             -nodesktop \
#             -noFigureWindows \
#             -nosplash \
#             -r \
#             {addpath}')


t1_list_preproc = [] # list of preprocessed T1: brain extracted, white / grey matter extracted, ANTs preprocessed
mask_list = []

for subj_preproc,subj_name in zip(t1_list,subjects):


    subj_folder = os.path.join(path_subj,subj_name)

    # create masks
    mask_pieces = ['l-pall','l-wm','l-caud','l-accu','l-thal','l-puta','r-pall','r-wm','r-caud','r-accu','r-thal','r-puta']
    rm_mask = ['wmh']

    start = os.path.join(subj_folder,mask_pieces[0]+'.nii.gz')
    for maskpiece in mask_pieces[1::]:
        next = os.path.join(subj_folder,maskpiece+'.nii.gz')
        output_mask = os.path.join(subj_folder,'wmdgmask.nii.gz')
        #combine
        os.system(f'{qit} MaskUnion \
              --left {start} \
              --right {next} \
              --output {output_mask}')
        start = output_mask
   

    #fill
    # step 4
    close_mask = os.path.join(subj_folder,'closed-mask.nii.gz')
    os.system(f'{qit} MaskClose \
              --input {output_mask} \
              --num {1} \
              --output {close_mask}')


    #subtract out wmh
    wmh_mask = os.path.join(subj_folder,rm_mask[0]+'.nii.gz')
    final_mask = os.path.join(subj_folder,subj_name+'-final_mask.nii.gz')
    os.system(f'{qit} MaskSet \
                  --input {output_mask} \
                  --mask {wmh_mask} \
                  --label {0} \
                  --output {final_mask}')
    
    # binarize
    os.system(f'{qit} MaskBinarize \
              --input {final_mask} \
              --output {final_mask}')



    mask_list.append(final_mask)
    print('mask done!')

    print('preproc done for '+subj_name+'!')

    #sys.exit()


#print(t1_list_preproc)



#print('subj list: '+subjects)
#t1_list_preproc = ['/Users/nanatang/Documents/GradResearch/ADNI3_samples/sub-4-ad/sub-4-ad-betbrain.nii.gz',\
                    # '/Users/nanatang/Documents/GradResearch/ADNI3_samples/sub-1-ad/sub-1-ad-betbrain.nii.gz', \
                    # '/Users/nanatang/Documents/GradResearch/ADNI3_samples/sub-5-ad/sub-5-ad-betbrain.nii.gz',\
                    # '/Users/nanatang/Documents/GradResearch/ADNI3_samples/sub-10-cn/sub-10-cn-betbrain.nii.gz',\
                    # '/Users/nanatang/Documents/GradResearch/ADNI3_samples/sub-11-cn/sub-11-cn-betbrain.nii.gz',\
                    # '/Users/nanatang/Documents/GradResearch/ADNI3_samples/sub-7-cn/sub-7-cn-betbrain.nii.gz',\
                    # '/Users/nanatang/Documents/GradResearch/ADNI3_samples/sub-12-cn/sub-12-cn-betbrain.nii.gz', \
                    # '/Users/nanatang/Documents/GradResearch/ADNI3_samples/sub-2-ad/sub-2-ad-betbrain.nii.gz',\
                    # '/Users/nanatang/Documents/GradResearch/ADNI3_samples/sub-6-ad/sub-6-ad-betbrain.nii.gz',\
                    # '/Users/nanatang/Documents/GradResearch/ADNI3_samples/sub-9-cn/sub-9-cn-betbrain.nii.gz',\
                    # '/Users/nanatang/Documents/GradResearch/ADNI3_samples/sub-8-cn/sub-8-cn-betbrain.nii.gz', \
                    # '/Users/nanatang/Documents/GradResearch/ADNI3_samples/sub-3-ad/sub-3-ad-betbrain.nii.gz']

# mask_list = ['/Users/nanatang/Documents/GradResearch/ADNI3_samples/sub-4-ad/sub-4-ad-allmask-closed.nii.gz',\
#                     '/Users/nanatang/Documents/GradResearch/ADNI3_samples/sub-1-ad/sub-1-ad-allmask-closed.nii.gz', \
#                     '/Users/nanatang/Documents/GradResearch/ADNI3_samples/sub-5-ad/sub-5-ad-allmask-closed.nii.gz',\
#                     '/Users/nanatang/Documents/GradResearch/ADNI3_samples/sub-10-cn/sub-10-cn-allmask-closed.nii.gz',\
#                     '/Users/nanatang/Documents/GradResearch/ADNI3_samples/sub-11-cn/sub-11-cn-allmask-closed.nii.gz',\
#                     '/Users/nanatang/Documents/GradResearch/ADNI3_samples/sub-7-cn/sub-7-cn-allmask-closed.nii.gz', \
#                     '/Users/nanatang/Documents/GradResearch/ADNI3_samples/sub-12-cn/sub-12-cn-allmask-closed.nii.gz', \
#                     '/Users/nanatang/Documents/GradResearch/ADNI3_samples/sub-2-ad/sub-2-ad-allmask-closed.nii.gz',\
#                     '/Users/nanatang/Documents/GradResearch/ADNI3_samples/sub-6-ad/sub-6-ad-allmask-closed.nii.gz',\
#                     '/Users/nanatang/Documents/GradResearch/ADNI3_samples/sub-9-cn/sub-9-cn-allmask-closed.nii.gz',\
#                     '/Users/nanatang/Documents/GradResearch/ADNI3_samples/sub-8-cn/sub-8-cn-allmask-closed.nii.gz',\
#                     '/Users/nanatang/Documents/GradResearch/ADNI3_samples/sub-3-ad/sub-3-ad-allmask-closed.nii.gz']

# gt_list = ['/Users/nanatang/Documents/GradResearch/frangi-valdo/sub-4-ad/sub-101_space-T1_desc-Rater1_PVSSeg.nii.gz',\
#                     '/Users/nanatang/Documents/GradResearch/frangi-valdo/sub-1-frangi/sub-8_space-T1_desc-Rater1_PVSSeg.nii.gz', \
#                     '/Users/nanatang/Documents/GradResearch/frangi-valdo/sub-105-frangi/sub-105_space-T1_desc-Rater1_PVSSeg.nii.gz',\
#                     '/Users/nanatang/Documents/GradResearch/frangi-valdo/sub-102-frangi/sub-102_space-T1_desc-Rater1_PVSSeg.nii.gz',\
#                     '/Users/nanatang/Documents/GradResearch/frangi-valdo/sub-103-frangi/sub-103_space-T1_desc-Rater1_PVSSeg.nii.gz',\
#                     '/Users/nanatang/Documents/GradResearch/frangi-valdo/sub-104-frangi/sub-104_space-T1_desc-Rater1_PVSSeg.nii.gz']

#for subj,m,wmh,gt in zip(t1_list_preproc,mask_list,wmh_list,gt_list):

#sys.exit()
ad_count_list = []
ad_vol_list = []
cn_count_list = []
cn_vol_list = []

for subj,m,subj_name in zip(t1_list,mask_list,subjects):

    #subject name
    #subj_name = subj.split('/')[7][:7]
    #subj_folder = subj.split('/'+subj_name+'-bnpreproc',1)[0]
    #subj_folder = '/Users/nanatang/Documents/GradResearch/frangi-valdo/'+subj_name+'-frangi'
    subj_folder = os.path.join(path_subj,subj_name)


    #print(subj_name)

    # hessian calculation
    hes =  os.path.join(subj_folder,subj_name+'-hessian.nii.gz')
    os.system(f'{qit} VolumeFilterHessian \
              --input {subj} \
              --mask {m} \
              --mode Norm \
              --output {hes}')

    hes_stats = os.path.join(subj_folder,subj_name+'-hessianstats.csv')
    os.system(f'{qit} VolumeMeasure \
              --input {hes} \
              --output {hes_stats}')

    hes_csv = pd.read_csv(hes_stats,index_col=0)
    half_max = hes_csv.loc['max'][0]/2


    # frangi calculation
    frangi_mask = os.path.join(subj_folder,subj_name+'-frangimask.nii.gz')
    os.system(f'{qit} VolumeFilterFrangi \
              --input {subj} \
              --mask {m} \
              --low {0.1} \
              --high {5.0} \
              --scales {10} \
              --gamma {half_max} \
              --dark \
              --output {frangi_mask}')


    ############# insert IQR scaling



    # threshold calculation
    # for now, set threshold to optimal threshold you found with sub-101 with bias / denoise 
    t = 0.0025

    frangi_thresholded = os.path.join(subj_folder,subj_name+'-frangimask-thresholded.nii.gz')
    os.system(f'{qit} VolumeThreshold \
              --input {frangi_mask} \
              --mask {m} \
              --threshold {t} \
              --output {frangi_thresholded}')

    print('subject '+ subj_name + ' frangi done!')


    # #######-----------remove WMH------------#########

    # frangi_wmhremoved = os.path.join(subj_folder, subj_name+'-frangimask-thresholded-wmhremoved.nii.gz')
    # os.system(f'{qit} MaskSet \
    #           --input {frangi_thresholded} \
    #           --mask {wmh} \
    #           --label {0} \
    #           --output {frangi_wmhremoved}')

    # #######-----------remove WMH------------#########




    # dice score
    #dice_subj = os.path.join(subj_folder, subj_name+'-dicescore.csv')

    # pvs count
    pvsseg_comp = os.path.join(subj_folder, subj_name+'-frangi_comp.nii.gz')
    # gt_comp = os.path.join(subj_folder, subj_name+'-groundtruth_comp.csv')
    pvsseg_stats = os.path.join(subj_folder, subj_name+'-pvsseg_stats.csv')
    #gt_stats = os.path.join(subj_folder, subj_name+'-groundtruth_stats.csv')

    # os.system(f'{qit} MaskDiceBatch \
    #           --left {frangi_thresholded} \
    #           --right {gt} \
    #           --output {dice_subj}')

    # # with wmh removal:
    # os.system(f'{qit} MaskDiceBatch \
    #       --left {frangi_wmhremoved} \
    #       --right {gt} \
    #       --output {dice_subj}')


    # mask component first
    os.system(f'{qit} MaskComponents \
          --input {frangi_thresholded} \
          --output {pvsseg_comp}')
    # os.system(f'{qit} MaskComponents \
    #       --input {gt} \
    #       --output {gt_comp}')
    

    # count volume and number of pvs
    os.system(f'{qit} MaskMeasure \
          --input {pvsseg_comp} \
          --comps \
          --counts \
          --output {pvsseg_stats}')

    # make list
    stat = pd.read_csv(pvsseg_stats,index_col=0)
    numpvs =  stat.loc['component_count'][0]
    volpvs = stat.loc['component_sum'][0]

    if "-ad" in subj_name:
        ad_vol_list.append(volpvs)
        ad_count_list.append(numpvs)
    else:
        cn_vol_list.append(volpvs)
        cn_count_list.append(numpvs)


ad_cases = np.arange(len(ad_vol_list))+1
cn_cases = np.arange(len(ad_vol_list))+11

ad_casesmean = np.mean(ad_cases)
cn_casesmean = np.mean(cn_cases)

ad_vol_listmean = np.mean(ad_vol_list)
cn_vol_listmean = np.mean(cn_vol_list)

ad_count_listmean = np.mean(ad_count_list)
cn_count_listmean = np.mean(cn_count_list)

ham = stats.ttest_ind(ad_vol_list,cn_vol_list).pvalue
bao = stats.ttest_ind(ad_count_list,cn_count_list).pvalue

fig_vol = plt.figure(figsize=(10,5))
plt.scatter(ad_cases,ad_vol_list,c='red',label='AD')
plt.scatter(ad_cases,cn_vol_list,c='green',label='CN')
plt.axhline(ad_vol_listmean,c='pink',label='AD mean')
plt.axhline(cn_vol_listmean,c='cyan',label='CN mean')
plt.title('AD vs CN PVS volume')
plt.legend()
plt.text(1,np.max(ad_vol_list)-np.min(cn_vol_list)+2,str(ham))
plt.show()
fig_vol.savefig(os.path.join(path_subj,'ADvsCN_pvsvol.jpeg'))

fig_count = plt.figure(figsize=(10,5))
plt.scatter(ad_cases,ad_count_list,c='red',label='AD')
plt.scatter(ad_cases,cn_count_list,c='green',label='CN')
plt.axhline(ad_count_listmean,c='pink',label='AD mean')
plt.axhline(cn_count_listmean,c='cyan',label='CN mean')
plt.title('AD vs CN PVS count')
plt.text(1,np.max(ad_count_list)-np.min(cn_count_list)+2,str(bao))
plt.legend()
plt.show()
fig_count.savefig(os.path.join(path_subj,'ADvsCN_pvscount.jpeg'))

np.append(ad_cases,ad_casesmean)
np.append(cn_cases,cn_casesmean)
ad_vol_list.append(ad_vol_listmean)
cn_vol_list.append(cn_vol_listmean)
ad_count_list.append(ad_count_listmean)
cn_count_list.append(cn_count_listmean)


alldata_advol = pd.DataFrame(data=zip(ad_vol_list),index=ad_cases,columns =['ad volume']).rename_axis('subjects')
alldata_cnvol = pd.DataFrame(data=zip(cn_vol_list),index=cn_cases,columns =['cn volume']).rename_axis('subjects')

alldata_adcount = pd.DataFrame(data=zip(ad_count_list),index=ad_cases,columns =['ad count']).rename_axis('subjects')
alldata_cncount = pd.DataFrame(data=zip(cn_count_list),index=cn_cases,columns =['cn count']).rename_axis('subjects')

alldata_advol.to_csv(os.path.join(path_subj,'ADvolumes.csv'), index=True)
alldata_cnvol.to_csv(os.path.join(path_subj,'CNvolumes.csv'), index=True)
alldata_adcount.to_csv(os.path.join(path_subj,'ADcounts.csv'), index=True)
alldata_cncount.to_csv(os.path.join(path_subj,'CNcounts.csv'), index=True)
