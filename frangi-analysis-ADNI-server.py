import os
import numpy as np
import shutil
import pandas as pd
import sys
import gzip
import nibabel as nib
from matplotlib import pyplot as plt
from scipy import stats

## catch errors, make error message


### commands ###
#qit command
qit = '/opt/qit/bin/qit'

#ANTs commands
bias_command = '/opt/ants/bin/N4BiasFieldCorrection'
denoise_command = '/opt/ants/bin/DenoiseImage'

#FSL commands
fsl_segment = '/opt/fsl/bin/fast'
fsl_brain = '/opt/fsl/bin/bet'
fsl_deepgrey = '/opt/fsl/bin/run_first_all'
fsl_firstflirt = '/opt/fsl/bin/first_flirt'
fsl_runfirst = '/opt/fsl/bin/run_first'
fsl_maths = '/opt/fsl/bin/fslmaths'


# #LST commands, addpath
# matlab = '/opt/mathworks/bin/matlab'
# addpath = "\"addpath('/m/Researchers/SerenaT/spm12');exit\""
# os.system(f'{matlab} \
#             -nodesktop \
#             -noFigureWindows \
#             -nosplash \
#             -r \
#             {addpath}')


#freesurfer commands
fs_mriconvert = '/opt/freesurfer/bin/mri_convert'
fs_asegtable = '/opt/freesurfer/bin/asegstats2table'




path_subj = '/Users/nanatang/VAserversim/m/Researchers/SerenaT/ADNI_samples'
ad_folder = os.path.join(path_subj,'AD')
cn_folder = os.path.join(path_subj,'CN')

ad_folder_ls = os.listdir(ad_folder)
cn_folder_ls = os.listdir(cn_folder)

# the "i for i in __ if i,startswith(___) is only meant to list out the folders and get rid of things that aren't folders you want (like .DStore)"

subject_id_ad = [i for i in ad_folder_ls if i.startswith('ADNI')]
subject_id_cn = [c for c in cn_folder_ls if c.startswith('ADNI')]


subjects_t1_ad = []
subjects_t1_cn = []
subject_mask_ad = []
subject_mask_cn = []

for a in subject_id_ad:
    t1 = os.path.join(path_subj,ad_folder,a,'T1.mgz')
    t1_convert = os.path.join(path_subj,ad_folder,a,'T1.nii.gz')
    os.system(f'{fs_mriconvert}  \
                {t1} \
                {t1_convert}')

    subjects_t1_ad.append(t1_convert)

    mask = os.path.join(path_subj,ad_folder,a,'aparc+aseg.mgz')
    subject_mask_ad.append(mask)

for c in subject_id_cn:
    t1 = os.path.join(path_subj,cn_folder,c,'T1.mgz')
    t1_convert = os.path.join(path_subj,cn_folder,c,'T1.nii.gz')
    os.system(f'{fs_mriconvert}  \
                {t1} \
                {t1_convert}')
    
    subjects_t1_cn.append(t1_convert)

    mask = os.path.join(path_subj,cn_folder,c,'aparc+aseg.mgz')
    subject_mask_cn.append(mask)

t1_list = subjects_t1_ad+subjects_t1_cn
mask_list = subject_mask_ad+subject_mask_cn



t1_list_preproc = [] # list of preprocessed T1: brain extracted, white / grey matter extracted, ANTs preprocessed
finalmask_list = []

# 8/22/23: new add - regional results
# wm_list = []
# bg_list = []

for maskmgz in mask_list:

    subj_folder = '/'+'/'.join(str.split(maskmgz,'/')[1:-1])

    # make mask with: white matter + deep grey + no WMH (according to fs) --> still need to incorporate WMH from SPM

    img = nib.load(maskmgz)
    data = img.get_fdata()
    mask = np.zeros(np.shape(data))

    seg = [2,10,11,12,13,26,41,49,50,51,52,58]
    wmh = 77    # do we need this if it should already not be included in the others? or is it included? or maybe it doesn't matter since 
                # she wants me to use SPM anyway

    for m in seg:
        mask[data == m] = m

    maskimg = nib.Nifti1Image(mask,img.affine)
    nib.save(maskimg,os.path.join(subj_folder,'wmdg-mask.nii.gz'))
    output_mask = os.path.join(subj_folder,'wmdg-mask.nii.gz')

   
    #fill
    # step 4
    final_mask = os.path.join(subj_folder,'closed-mask.nii.gz')
    os.system(f'{qit} MaskClose \
              --input {output_mask} \
              --num {1} \
              --output {final_mask}')

    
    # binarize
    os.system(f'{qit} MaskBinarize \
              --input {final_mask} \
              --output {final_mask}')


    finalmask_list.append(final_mask)

    # ####--------wm mask---------######
    # wmmask = os.path.join(subj_folder,subj_name+'-wmmask.nii.gz')
    # lwm = os.path.join(subj_folder,'l-wm.nii.gz')
    # rwm = os.path.join(subj_folder,'r-wm.nii.gz')
    # os.system(f'{qit} MaskUnion \
    #           --left {lwm} \
    #           --right {rwm} \
    #           --output {wmmask}')
    # os.system(f'{qit} MaskClose \
    #           --input {wmmask} \
    #           --num {1} \
    #           --output {wmmask}')

    # wm_list.append(wmmask)


    # ####--------bg mask---------######
    # bgmask = os.path.join(subj_folder,subj_name+'-bgmask.nii.gz')
    # os.system(f'{qit} MaskSet \
    #               --input {final_mask} \
    #               --mask {wmmask} \
    #               --label {0} \
    #               --output {bgmask}')

    
    # bg_list.append(bgmask)
    

    print('masks done!')

    print('preproc done for '+str.split(maskmgz,'/')[-2]+'!')

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

wm_ad_count_list = []
wm_ad_vol_list = []
wm_cn_count_list = []
wm_cn_vol_list = []

bg_ad_count_list = []
bg_ad_vol_list = []
bg_cn_count_list = []
bg_cn_vol_list = []

# for subj,m,m_wm,m_bg,subj_name in zip(t1_list,mask_list,wm_list,bg_list,subjects):
for subj,m in zip(t1_list,finalmask_list):

 
    subj_folder = '/'+'/'.join(str.split(subj,'/')[1:-1])

    ####------total brain calculation------#####

    # hessian calculation
    hes =  os.path.join(subj_folder,'hessian.nii.gz')
    os.system(f'{qit} VolumeFilterHessian \
              --input {subj} \
              --mask {m} \
              --mode Norm \
              --output {hes}')

    hes_stats = os.path.join(subj_folder,'hessianstats.csv')
    os.system(f'{qit} VolumeMeasure \
              --input {hes} \
              --output {hes_stats}')

    hes_csv = pd.read_csv(hes_stats,index_col=0)
    half_max = hes_csv.loc['max'][0]/2


    # frangi calculation
    frangi_mask = os.path.join(subj_folder,'frangimask.nii.gz')
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

    frangi_thresholded = os.path.join(subj_folder,'frangimask-thresholded.nii.gz')
    os.system(f'{qit} VolumeThreshold \
              --input {frangi_mask} \
              --mask {m} \
              --threshold {t} \
              --output {frangi_thresholded}')

    print('subject '+ str.split(subj,'/')[-2] + ' total frangi done!')


    ###########-------stats calculation------#################    

    # pvs count
    pvsseg_comp = os.path.join(subj_folder, 'frangi_comp.nii.gz')
    # gt_comp = os.path.join(subj_folder, subj_name+'-groundtruth_comp.csv')
    pvsseg_stats = os.path.join(subj_folder, 'pvsseg_stats.csv')
    #gt_stats = os.path.join(subj_folder, subj_name+'-groundtruth_stats.csv')

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

    # # ICV calculation - what do I do with this?
    # aseg_raw = os.path.join(subj_folder,'aseg.stats')
    # aseg_file = os.path.join(subj_folder,'asegstats.csv')

    # os.system(f'{fs_asegtable} \
    #             -i {aseg_raw} \
    #             -d comma \
    #             -t {aseg_file}')
    
    # stat = pd.read_csv(aseg_file)
    # icv = stat['EstimatedTotalIntrCranialVol'][0]




    # if "-ad" in subj_name:
    #     ad_vol_list.append(volpvs)
    #     ad_count_list.append(numpvs)
    # else:
    #     cn_vol_list.append(volpvs)
    #     cn_count_list.append(numpvs)


    # ####------wm brain calculation------#####

    # # hessian calculation
    # hes =  os.path.join(subj_folder,subj_name+'-hessian_wm.nii.gz')
    # os.system(f'{qit} VolumeFilterHessian \
    #           --input {subj} \
    #           --mask {m_wm} \
    #           --mode Norm \
    #           --output {hes}')

    # hes_stats = os.path.join(subj_folder,subj_name+'-hessianstats_wm.csv')
    # os.system(f'{qit} VolumeMeasure \
    #           --input {hes} \
    #           --output {hes_stats}')

    # hes_csv = pd.read_csv(hes_stats,index_col=0)
    # half_max = hes_csv.loc['max'][0]/2


    # # frangi calculation
    # frangi_mask = os.path.join(subj_folder,subj_name+'-frangimask_wmwm.nii.gz')
    # os.system(f'{qit} VolumeFilterFrangi \
    #           --input {subj} \
    #           --mask {m_wm} \
    #           --low {0.1} \
    #           --high {5.0} \
    #           --scales {10} \
    #           --gamma {half_max} \
    #           --dark \
    #           --output {frangi_mask}')


    # ############# insert IQR scaling


    # # threshold calculation
    # # for now, set threshold to optimal threshold you found with sub-101 with bias / denoise 
    # t = 0.0025

    # frangi_thresholded = os.path.join(subj_folder,subj_name+'-frangimask-thresholded_wm.nii.gz')
    # os.system(f'{qit} VolumeThreshold \
    #           --input {frangi_mask} \
    #           --mask {m_wm} \
    #           --threshold {t} \
    #           --output {frangi_thresholded}')

    # print('subject '+ subj_name + ' wm frangi done!')


    # # pvs count
    # pvsseg_comp = os.path.join(subj_folder, subj_name+'-frangi_comp_wm.nii.gz')
    # # gt_comp = os.path.join(subj_folder, subj_name+'-groundtruth_comp.csv')
    # pvsseg_stats = os.path.join(subj_folder, subj_name+'-pvsseg_stats_wm.csv')
    # #gt_stats = os.path.join(subj_folder, subj_name+'-groundtruth_stats.csv')

    # # mask component first
    # os.system(f'{qit} MaskComponents \
    #       --input {frangi_thresholded} \
    #       --output {pvsseg_comp}')
    # # os.system(f'{qit} MaskComponents \
    # #       --input {gt} \
    # #       --output {gt_comp}')
    

    # # count volume and number of pvs
    # os.system(f'{qit} MaskMeasure \
    #       --input {pvsseg_comp} \
    #       --comps \
    #       --counts \
    #       --output {pvsseg_stats}')

    # # make list
    # stat = pd.read_csv(pvsseg_stats,index_col=0)
    # numpvs =  stat.loc['component_count'][0]
    # volpvs = stat.loc['component_sum'][0]

    # if "-ad" in subj_name:
    #     wm_ad_vol_list.append(volpvs)
    #     wm_ad_count_list.append(numpvs)
    # else:
    #     wm_cn_vol_list.append(volpvs)
    #     wm_cn_count_list.append(numpvs)


    # ####------bg brain calculation------#####

    # # hessian calculation
    # hes =  os.path.join(subj_folder,subj_name+'-hessian-bg.nii.gz')
    # os.system(f'{qit} VolumeFilterHessian \
    #           --input {subj} \
    #           --mask {m_bg} \
    #           --mode Norm \
    #           --output {hes}')

    # hes_stats = os.path.join(subj_folder,subj_name+'-hessianstats-bg.csv')
    # os.system(f'{qit} VolumeMeasure \
    #           --input {hes} \
    #           --output {hes_stats}')

    # hes_csv = pd.read_csv(hes_stats,index_col=0)
    # half_max = hes_csv.loc['max'][0]/2


    # # frangi calculation
    # frangi_mask = os.path.join(subj_folder,subj_name+'-frangimask-bg.nii.gz')
    # os.system(f'{qit} VolumeFilterFrangi \
    #           --input {subj} \
    #           --mask {m_bg} \
    #           --low {0.1} \
    #           --high {5.0} \
    #           --scales {10} \
    #           --gamma {half_max} \
    #           --dark \
    #           --output {frangi_mask}')


    # ############# insert IQR scaling


    # # threshold calculation
    # # for now, set threshold to optimal threshold you found with sub-101 with bias / denoise 
    # t = 0.0025

    # frangi_thresholded = os.path.join(subj_folder,subj_name+'-frangimask-thresholded.nii.gz')
    # os.system(f'{qit} VolumeThreshold \
    #           --input {frangi_mask} \
    #           --mask {m_bg} \
    #           --threshold {t} \
    #           --output {frangi_thresholded}')

    # print('subject '+ subj_name + ' bg frangi done!')


    # # pvs count
    # pvsseg_comp = os.path.join(subj_folder, subj_name+'-frangi_comp_bg.nii.gz')
    # # gt_comp = os.path.join(subj_folder, subj_name+'-groundtruth_comp.csv')
    # pvsseg_stats = os.path.join(subj_folder, subj_name+'-pvsseg_stats_bg.csv')
    # #gt_stats = os.path.join(subj_folder, subj_name+'-groundtruth_stats.csv')

    # # mask component first
    # os.system(f'{qit} MaskComponents \
    #       --input {frangi_thresholded} \
    #       --output {pvsseg_comp}')
    # # os.system(f'{qit} MaskComponents \
    # #       --input {gt} \
    # #       --output {gt_comp}')
    

    # # count volume and number of pvs
    # os.system(f'{qit} MaskMeasure \
    #       --input {pvsseg_comp} \
    #       --comps \
    #       --counts \
    #       --output {pvsseg_stats}')

    # # make list
    # stat = pd.read_csv(pvsseg_stats,index_col=0)
    # numpvs =  stat.loc['component_count'][0]
    # volpvs = stat.loc['component_sum'][0]

    # if "-ad" in subj_name:
    #     bg_ad_vol_list.append(volpvs)
    #     bg_ad_count_list.append(numpvs)
    # else:
    #     bg_cn_vol_list.append(volpvs)
    #     bg_cn_count_list.append(numpvs)



# ####-----total brain plots-----####

# ad_cases = np.arange(len(ad_vol_list))+1
# cn_cases = np.arange(len(ad_vol_list))+11

# ad_casesmean = np.mean(ad_cases)
# cn_casesmean = np.mean(cn_cases)

# ad_vol_listmean = np.mean(ad_vol_list)
# cn_vol_listmean = np.mean(cn_vol_list)

# ad_count_listmean = np.mean(ad_count_list)
# cn_count_listmean = np.mean(cn_count_list)

# ham = stats.ttest_ind(ad_vol_list,cn_vol_list).pvalue
# bao = stats.ttest_ind(ad_count_list,cn_count_list).pvalue

# fig_vol = plt.figure(figsize=(10,5))
# plt.scatter(ad_cases,ad_vol_list,c='red',label='AD')
# plt.scatter(ad_cases,cn_vol_list,c='green',label='CN')
# plt.axhline(ad_vol_listmean,c='pink',label='AD mean')
# plt.axhline(cn_vol_listmean,c='cyan',label='CN mean')
# plt.title('AD vs CN PVS volume')
# plt.legend()
# plt.text(1,np.max(ad_vol_list)-np.min(cn_vol_list)+2,str(ham))
# plt.show()
# fig_vol.savefig(os.path.join(path_subj,'ADvsCN_pvsvol.jpeg'))

# fig_count = plt.figure(figsize=(10,5))
# plt.scatter(ad_cases,ad_count_list,c='red',label='AD')
# plt.scatter(ad_cases,cn_count_list,c='green',label='CN')
# plt.axhline(ad_count_listmean,c='pink',label='AD mean')
# plt.axhline(cn_count_listmean,c='cyan',label='CN mean')
# plt.title('AD vs CN PVS count')
# plt.text(1,np.max(ad_count_list)-np.min(cn_count_list)+2,str(bao))
# plt.legend()
# plt.show()
# fig_count.savefig(os.path.join(path_subj,'ADvsCN_pvscount.jpeg'))

# ad_cases = np.append(ad_cases,ad_casesmean)
# cn_cases = np.append(cn_cases,cn_casesmean)
# ad_vol_list.append(ad_vol_listmean)
# cn_vol_list.append(cn_vol_listmean)
# ad_count_list.append(ad_count_listmean)
# cn_count_list.append(cn_count_listmean)


# alldata_advol = pd.DataFrame(data=zip(ad_vol_list),index=ad_cases,columns =['ad volume']).rename_axis('subjects')
# alldata_cnvol = pd.DataFrame(data=zip(cn_vol_list),index=cn_cases,columns =['cn volume']).rename_axis('subjects')

# alldata_adcount = pd.DataFrame(data=zip(ad_count_list),index=ad_cases,columns =['ad count']).rename_axis('subjects')
# alldata_cncount = pd.DataFrame(data=zip(cn_count_list),index=cn_cases,columns =['cn count']).rename_axis('subjects')

# alldata_advol.to_csv(os.path.join(path_subj,'ADvolumes.csv'), index=True)
# alldata_cnvol.to_csv(os.path.join(path_subj,'CNvolumes.csv'), index=True)
# alldata_adcount.to_csv(os.path.join(path_subj,'ADcounts.csv'), index=True)
# alldata_cncount.to_csv(os.path.join(path_subj,'CNcounts.csv'), index=True)



# # ####-----wm brain plots-----####

# # ad_cases = np.arange(len(ad_vol_list))+1
# # cn_cases = np.arange(len(ad_vol_list))+11

# # wm_ad_vol_listmean = np.mean(wm_ad_vol_list)
# # wm_cn_vol_listmean = np.mean(wm_cn_vol_list)

# # wm_ad_count_listmean = np.mean(wm_ad_count_list)
# # wm_cn_count_listmean = np.mean(wm_cn_count_list)

# # ham = stats.ttest_ind(wm_ad_vol_list,wm_cn_vol_list).pvalue
# # bao = stats.ttest_ind(wm_ad_count_list,wm_cn_count_list).pvalue

# # wm_fig_vol = plt.figure(figsize=(10,5))
# # plt.scatter(ad_cases,wm_ad_vol_list,c='red',label='AD')
# # plt.scatter(ad_cases,wm_cn_vol_list,c='green',label='CN')
# # plt.axhline(wm_ad_vol_listmean,c='pink',label='AD mean')
# # plt.axhline(wm_cn_vol_listmean,c='cyan',label='CN mean')
# # plt.title('AD vs CN PVS volume (white matter)')
# # plt.legend()
# # plt.text(1,np.max(wm_ad_vol_list)-np.min(wm_cn_vol_list)+2,str(ham))
# # plt.show()
# # wm_fig_vol.savefig(os.path.join(path_subj,'ADvsCN_pvsvol_wm.jpeg'))

# # wm_fig_count = plt.figure(figsize=(10,5))
# # plt.scatter(ad_cases,wm_ad_count_list,c='red',label='AD')
# # plt.scatter(ad_cases,wm_cn_count_list,c='green',label='CN')
# # plt.axhline(wm_ad_count_listmean,c='pink',label='AD mean')
# # plt.axhline(wm_cn_count_listmean,c='cyan',label='CN mean')
# # plt.title('AD vs CN PVS count (white matter)')
# # plt.text(1,np.max(wm_ad_count_list)-np.min(wm_cn_count_list)+2,str(bao))
# # plt.legend()
# # plt.show()
# # wm_fig_count.savefig(os.path.join(path_subj,'ADvsCN_pvscount_wm.jpeg'))

# # ad_cases = np.append(ad_cases,ad_casesmean)
# # cn_cases = np.append(cn_cases,cn_casesmean)
# # wm_ad_vol_list.append(wm_ad_vol_listmean)
# # wm_cn_vol_list.append(wm_cn_vol_listmean)
# # wm_ad_count_list.append(wm_ad_count_listmean)
# # wm_cn_count_list.append(wm_cn_count_listmean)


# # alldata_advol = pd.DataFrame(data=zip(wm_ad_vol_list),index=ad_cases,columns =['ad volume,wm']).rename_axis('subjects')
# # alldata_cnvol = pd.DataFrame(data=zip(wm_cn_vol_list),index=cn_cases,columns =['cn volume,wm']).rename_axis('subjects')

# # alldata_adcount = pd.DataFrame(data=zip(ad_count_list),index=ad_cases,columns =['ad count,bg']).rename_axis('subjects')
# # alldata_cncount = pd.DataFrame(data=zip(cn_count_list),index=cn_cases,columns =['cn count,bg']).rename_axis('subjects')

# # alldata_advol.to_csv(os.path.join(path_subj,'ADvolumes_wm.csv'), index=True)
# # alldata_cnvol.to_csv(os.path.join(path_subj,'CNvolumes_wm.csv'), index=True)
# # alldata_adcount.to_csv(os.path.join(path_subj,'ADcounts_wm.csv'), index=True)
# # alldata_cncount.to_csv(os.path.join(path_subj,'CNcounts_wm.csv'), index=True)





# # ####-----bg brain plots-----####

# # ad_cases = np.arange(len(ad_vol_list))+1
# # cn_cases = np.arange(len(ad_vol_list))+11


# # bg_ad_vol_listmean = np.mean(bg_ad_vol_list)
# # bg_cn_vol_listmean = np.mean(bg_cn_vol_list)

# # bg_ad_count_listmean = np.mean(bg_ad_count_list)
# # bg_cn_count_listmean = np.mean(bg_cn_count_list)

# # ham = stats.ttest_ind(bg_ad_vol_list,bg_cn_vol_list).pvalue
# # bao = stats.ttest_ind(bg_ad_count_list,bg_cn_count_list).pvalue

# # bg_fig_vol = plt.figure(figsize=(10,5))
# # plt.scatter(ad_cases,bg_ad_vol_list,c='red',label='AD')
# # plt.scatter(ad_cases,bg_cn_vol_list,c='green',label='CN')
# # plt.axhline(bg_ad_vol_listmean,c='pink',label='AD mean')
# # plt.axhline(bg_cn_vol_listmean,c='cyan',label='CN mean')
# # plt.title('AD vs CN PVS volume (basal ganglia)')
# # plt.legend()
# # plt.text(1,np.max(bg_ad_vol_list)-np.min(bg_cn_vol_list)+2,str(ham))
# # plt.show()
# # bg_fig_vol.savefig(os.path.join(path_subj,'ADvsCN_pvsvol_bg.jpeg'))

# # bg_fig_count = plt.figure(figsize=(10,5))
# # plt.scatter(ad_cases,bg_ad_count_list,c='red',label='AD')
# # plt.scatter(ad_cases,bg_cn_count_list,c='green',label='CN')
# # plt.axhline(bg_ad_count_listmean,c='pink',label='AD mean')
# # plt.axhline(bg_cn_count_listmean,c='cyan',label='CN mean')
# # plt.title('AD vs CN PVS count (basal ganglia)')
# # plt.text(1,np.max(bg_ad_count_list)-np.min(bg_cn_count_list)+2,str(bao))
# # plt.legend()
# # plt.show()
# # bg_fig_count.savefig(os.path.join(path_subj,'ADvsCN_pvscount_bg.jpeg'))

# # ad_cases = np.append(ad_cases,ad_casesmean)
# # cn_cases = np.append(cn_cases,cn_casesmean)
# # bg_ad_vol_list.append(bg_ad_vol_listmean)
# # bg_cn_vol_list.append(bg_cn_vol_listmean)
# # bg_ad_count_list.append(bg_ad_count_listmean)
# # bg_cn_count_list.append(bg_cn_count_listmean)


# # alldata_advol = pd.DataFrame(data=zip(bg_ad_vol_list),index=ad_cases,columns =['ad volume,bg']).rename_axis('subjects')
# # alldata_cnvol = pd.DataFrame(data=zip(bg_cn_vol_list),index=cn_cases,columns =['cn volume,bg']).rename_axis('subjects')

# # alldata_adcount = pd.DataFrame(data=zip(ad_count_list),index=ad_cases,columns =['ad count,bg']).rename_axis('subjects')
# # alldata_cncount = pd.DataFrame(data=zip(cn_count_list),index=cn_cases,columns =['cn count,bg']).rename_axis('subjects')

# # alldata_advol.to_csv(os.path.join(path_subj,'ADvolumes_bg.csv'), index=True)
# # alldata_cnvol.to_csv(os.path.join(path_subj,'CNvolumes_bg.csv'), index=True)
# # alldata_adcount.to_csv(os.path.join(path_subj,'ADcounts_bg.csv'), index=True)
# # alldata_cncount.to_csv(os.path.join(path_subj,'CNcounts_bg.csv'), index=True)

