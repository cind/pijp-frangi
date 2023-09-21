import os
import numpy as np
import shutil
import pandas as pd
import sys
import gzip
import nibabel as nib
from matplotlib import pyplot as plt
from scipy import stats


###---------------------------commands---------------------------###

#for server
#qit command
qit = '/opt/qit/bin/qit'

#ANTs commands
os.environ['ANTSVERSION'] = 'ants-2017-12-07'
bias_command = '/opt/ants/bin/N4BiasFieldCorrection'
denoise_command = '/opt/ants/bin/DenoiseImage'

#FSL commands
os.environ['FSLVERSION']='6.0.0' 
fsl_segment = '/opt/fsl/bin/fast'
fsl_brain = '/opt/fsl/bin/bet'
fsl_deepgrey = '/opt/fsl/bin/run_first_all'
fsl_firstflirt = '/opt/fsl/bin/first_flirt'
fsl_runfirst = '/opt/fsl/bin/run_first'
fsl_maths = '/opt/fsl/bin/fslmaths'


# #LST commands, addpath
# os.environ['MATLAB_VERSION']='R2019a' 
# matlab = '/opt/mathworks/bin/matlab'
# addpath = "\"addpath('/m/Researchers/SerenaT/spm12');exit\""
# os.system(f'{matlab} \
#             -nodesktop \
#             -noFigureWindows \
#             -nosplash \
#             -r \
#             {addpath}')


#freesurfer commands
os.environ['FSVERSION']='6.0.0' 
fs_mriconvert = '/opt/freesurfer/bin/mri_convert'
fs_asegtable = '/opt/freesurfer/bin/asegstats2table'

# for your computer
#qit command
#qit command
# qit = '/Users/nanatang/anaconda3/envs/pvssegment/bin/qit-build-mac-latest/bin/qit'

# #ANTs commands
# bias_command = '/Users/nanatang/anaconda3/envs/pvssegment/build/ANTS-build/Examples/N4BiasFieldCorrection'
# denoise_command = '/Users/nanatang/anaconda3/envs/pvssegment/build/ANTS-build/Examples/DenoiseImage'

# #FSL commands
# fsl_segment = '/Users/nanatang/Documents/GradResearch/fsl/bin/fast'
# fsl_brain = '/Users/nanatang/Documents/GradResearch/fsl/bin/bet'
# fsl_deepgrey = '/Users/nanatang/Documents/GradResearch/fsl/bin/run_first_all'
# fsl_firstflirt = '/Users/nanatang/Documents/GradResearch/fsl/bin/first_flirt'
# fsl_runfirst = '/Users/nanatang/Documents/GradResearch/fsl/bin/run_first'
# fsl_maths = '/Users/nanatang/Documents/GradResearch/fsl/bin/fslmaths'


# # #LST commands, addpath
# # os.environ['MATLAB_VERSION']='R2019a' 
# # matlab = '/opt/mathworks/bin/matlab'
# # addpath = "\"addpath('/m/Researchers/SerenaT/spm12');exit\""
# # os.system(f'{matlab} \
# #             -nodesktop \
# #             -noFigureWindows \
# #             -nosplash \
# #             -r \
# #             {addpath}')


# #freesurfer commands
# fs_mriconvert = '/Applications/freesurfer/7.3.2/bin/mri_convert'
# fs_asegtable = '/Applications/freesurfer/7.3.2/bin/asegstats2table'


####---------------------organizing data into AD/CN folders-------------------##########

# #for your computer: 
# data_dir = '/Users/nanatang/VAserversim/m/Researchers/SerenaT/ADNI_samples/'
# working_dir = '/Users/nanatang/VAserversim/m/Researchers/SerenaT/ADNI_samples/frangi'

# #for server:
data_dir = '/m/InProcess/External/ADNI_FSdn/Freesurfer/subjects/'
working_dir = '/m/InProcess/External/ADNI3/ADNI3_frangi/pijp-frangi'
researcher_dir = '/m/Researchers/SerenaT/ADNI3_files'

os.makedirs(working_dir,exist_ok=True)
os.makedirs(researcher_dir,exist_ok=True)
#data_dir = '/m/Researchers/SerenaT/ADNI3_samples'

ref_csv = os.path.join(researcher_dir,'ADNI3_T1_namefilter.csv')
ref_list = pd.read_csv(ref_csv)
folders = os.listdir(data_dir)
subjects = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir,d)) & d.startswith('ADNI')]
subjects = subjects[0:20]

# # make research group folders
os.makedirs(os.path.join(working_dir,'AD'),exist_ok=True)
os.makedirs(os.path.join(working_dir,'CN'),exist_ok=True)
os.makedirs(os.path.join(working_dir,'MCI'),exist_ok=True)

# # replace years with y0__
replacements = {'ADNI Screening':'y00','ADNI3 Year 1 Visit': 'y01', 'ADNI3 Year 2 Visit': 'y02','ADNI3 Year 3 Visit': 'y03','ADNI3 Year 4 Visit': 'y04'}
ref_list['Visit'] = ref_list['Visit'].map(replacements).fillna(ref_list['Visit'])


inds = []
researchgroup = []
t1_list = []
mask_list = []
wmparc_list = []
icv_list = []

def mgz_convert(mgz,nii):
   os.system(f'{fs_mriconvert}  \
                {mgz} \
                {nii}')

def aseg_convert(aseg_raw,aseg_file):
    os.system(f'{fs_asegtable} \
                -i {aseg_raw} \
                -d comma \
                -t {aseg_file}')

for subj in subjects:
    for id,visit in zip(ref_list['Subject ID'],ref_list['Visit']):
        if (id in subj) & (visit in subj):
            ind = ref_list.index[(ref_list['Subject ID']==id) & (ref_list['Visit'] == visit)][0]
            if ind in inds:
                pass
            else:
                inds.append(ind)
                rg = ref_list.loc[ind]['Research Group']
                researchgroup.append(rg)
                
                os.makedirs(os.path.join(working_dir,rg,subj),exist_ok=True)

                # # your computer
                # t1 = os.path.join(data_dir,subj,'T1.mgz')
                # t1_convert = os.path.join(working_dir,rg,subj,'T1.nii.gz')
                # mgz_convert(t1,t1_convert)
                # t1_list.append(t1_convert)

                # mask = os.path.join(data_dir,subj,'aparc+aseg.mgz')
                # mask_dest = os.path.join(working_dir,rg,subj,'aparc+aseg.mgz')
                # shutil.copy(mask,mask_dest)
                # mask_list.append(mask_dest)

                # wm = os.path.join(data_dir,subj,'wmparc.mgz')
                # wm_convert = os.path.join(working_dir,rg,subj,'wmparc.nii.gz')
                # mgz_convert(wm,wm_convert)
                # wmparc_list.append(wm_convert)

                # server
                t1 = os.path.join(data_dir,subj,'mri','T1.mgz')
                t1_convert = os.path.join(working_dir,rg,subj,'T1.nii.gz')
                mgz_convert(t1,t1_convert)
                t1_list.append(t1_convert)

                mask = os.path.join(data_dir,subj,'mri','aparc+aseg.mgz')
                mask_dest = os.path.join(working_dir,rg,subj,'aparc+aseg.mgz')
                shutil.copy(mask,mask_dest)
                mask_list.append(mask_dest)

                wm = os.path.join(data_dir,subj,'wmparc.mgz')
                wm_convert = os.path.join(working_dir,rg,subj,'wmparc.nii.gz')
                mgz_convert(wm,wm_convert)
                wmparc_list.append(wm_convert)

                icv = os.path.join(data_dir,subj,'stats','aseg.stats')
                icv_convert = os.path.join(working_dir,rg,subj,'asegstats.csv')
                aseg_convert(icv,icv_convert)
                icv_list.append(icv_convert)

                


##############------------------------------------make mask list------------------------################

finalmask_list = []

def make_mask(maskmgz,subj_folder):
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
    masknii = os.path.join(subj_folder,'wmdg-mask.nii.gz')

    return masknii


for maskmgz in mask_list:

    subj_folder = '/'+'/'.join(str.split(maskmgz,'/')[1:-1])

    # make mask with: white matter + deep grey + no WMH (according to fs) --> still need to incorporate WMH from SPM
    output_mask = make_mask(maskmgz,subj_folder)

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

    print('masks done!')

    print('preproc done for '+str.split(maskmgz,'/')[-2]+'!')

    #sys.exit()


#sys.exit()


##############------------------------------------frangi filtering------------------------################

def frangi_analysis(t1,mask,threshold,frangi_thresholded,subj_folder):
    # hessian calculation
    hes =  os.path.join(subj_folder,'hessian.nii.gz')
    os.system(f'{qit} VolumeFilterHessian \
              --input {t1} \
              --mask {mask} \
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
              --input {t1} \
              --mask {mask} \
              --low {0.1} \
              --high {5.0} \
              --scales {10} \
              --gamma {half_max} \
              --dark \
              --output {frangi_mask}')

    # threshold calculation

    os.system(f'{qit} VolumeThreshold \
              --input {frangi_mask} \
              --mask {mask} \
              --threshold {t} \
              --output {frangi_thresholded}')
    
final_frangi_list = []
final_frangi_list_wm = []
# for subj,m,m_wm,m_bg,subj_name in zip(t1_list,mask_list,wm_list,bg_list,subjects):
for subj,m,wmparc in zip(t1_list,finalmask_list,wmparc_list):

    subj_folder = '/'+'/'.join(str.split(subj,'/')[1:-1])

    ####------total brain calculation------#####
    frangi_thresholded = os.path.join(subj_folder,'frangimask-thresholded.nii.gz')
    frangi_analysis(subj,m,0.0025,frangi_thresholded,subj_folder)
    final_frangi_list.append(frangi_thresholded)

    frangi_thresholded_wm = os.path.join(subj_folder,'frangimask-thresholded_wm.nii.gz')
    frangi_analysis(subj,wmparc,0.0002,frangi_thresholded_wm,subj_folder)
    final_frangi_list_wm.append(frangi_thresholded_wm)

    print('subject '+ str.split(subj,'/')[-2] + ' total frangi done!')



##############------------------------------------stats calculation------------------------################


 
# icv_list = []
volpvs_list = []
volpvs_list_wm = []
countpvs_list = []
countpvs_list_wm = []
icv_norm_list = []
icv_norm_list_wm = []
researchgroup_list = []
subjname_list = []

def pvs_stats(frangi,comp,stats):
    # mask component first
    os.system(f'{qit} MaskComponents \
        --input {frangi} \
        --output {comp}')
    
    # count volume and number of pvs
    os.system(f'{qit} MaskMeasure \
        --input {comp} \
        --comps \
        --counts \
        --output {stats}')
    
def icv_calc(aseg):
    # return 0
    stat = pd.read_csv(aseg)
    icv = stat['EstimatedTotalIntrCranialVol'][0]
    return icv

for frangi,frangi_wm,icv_file in zip(final_frangi_list,final_frangi_list_wm,icv_list):
        
    subj_folder = '/'+'/'.join(str.split(frangi,'/')[1:-1])

    # pvs count
    pvsseg_comp = os.path.join(subj_folder, 'frangi_comp.nii.gz')
    pvsseg_stats = os.path.join(subj_folder, 'pvsseg_stats.csv')

    pvsseg_comp_wm = os.path.join(subj_folder, 'frangi_comp_wm.nii.gz')
    pvsseg_stats_wm = os.path.join(subj_folder, 'pvsseg_stats_wm.csv')

    pvs_stats(frangi,pvsseg_comp,pvsseg_stats)

    pvs_stats(frangi_wm,pvsseg_comp_wm,pvsseg_stats_wm)

    # make list
    stats = pd.read_csv(pvsseg_stats,index_col=0)
    countpvs =  stats.loc['component_count'][0]    # number of PVS counted
    volpvs = stats.loc['component_sum'][0]       # number of voxels

    stats_wm = pd.read_csv(pvsseg_stats_wm,index_col=0)
    countpvs_wm =  stats_wm.loc['component_count'][0]    # number of PVS counted
    volpvs_wm = stats_wm.loc['component_sum'][0]       # number of voxels


    # ICV calculation - what do I do with this?
    icv = icv_calc(icv_file)
    # calculate the normalized volume / voxels --> right now using 1 vox = 1mm^3
    volpvs_norm = volpvs/icv
    volpvs_norm_wm = volpvs_wm/icv


    volpvs_list.append(volpvs)
    countpvs_list.append(countpvs)
    volpvs_list_wm.append(volpvs_wm)
    countpvs_list_wm.append(countpvs_wm)
    icv_norm_list.append(volpvs_norm)
    icv_norm_list_wm.append(volpvs_norm_wm)
    subjname_list.append(str.split(frangi,'/')[-2])
    researchgroup_list.append(str.split(frangi,'/')[-3])


    print('subject '+ str.split(frangi,'/')[-2] + ' stats done!')



##############----------------------------------------------write all results to csv--------------------------------------################

col = ['subjects','research group','pvsvol','pvscount','pvsvolwm','pvscountwm','icv norm','icv norm wm']

alldata = pd.DataFrame(data=zip(subjname_list,researchgroup_list,volpvs_list,countpvs_list,volpvs_list_wm,countpvs_list_wm,icv_norm_list,icv_norm_list_wm),index=np.arange(len(subjname_list))+1,columns=col)

alldata.to_csv(os.path.join(working_dir,'frangidata.csv'), index=True)



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
# fig_vol.savefig(os.path.join(data_dir,'ADvsCN_pvsvol.jpeg'))

# fig_count = plt.figure(figsize=(10,5))
# plt.scatter(ad_cases,ad_count_list,c='red',label='AD')
# plt.scatter(ad_cases,cn_count_list,c='green',label='CN')
# plt.axhline(ad_count_listmean,c='pink',label='AD mean')
# plt.axhline(cn_count_listmean,c='cyan',label='CN mean')
# plt.title('AD vs CN PVS count')
# plt.text(1,np.max(ad_count_list)-np.min(cn_count_list)+2,str(bao))
# plt.legend()
# plt.show()
# fig_count.savefig(os.path.join(data_dir,'ADvsCN_pvscount.jpeg'))

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

# alldata_advol.to_csv(os.path.join(data_dir,'ADvolumes.csv'), index=True)
# alldata_cnvol.to_csv(os.path.join(data_dir,'CNvolumes.csv'), index=True)
# alldata_adcount.to_csv(os.path.join(data_dir,'ADcounts.csv'), index=True)
# alldata_cncount.to_csv(os.path.join(data_dir,'CNcounts.csv'), index=True)



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
# # wm_fig_vol.savefig(os.path.join(data_dir,'ADvsCN_pvsvol_wm.jpeg'))

# # wm_fig_count = plt.figure(figsize=(10,5))
# # plt.scatter(ad_cases,wm_ad_count_list,c='red',label='AD')
# # plt.scatter(ad_cases,wm_cn_count_list,c='green',label='CN')
# # plt.axhline(wm_ad_count_listmean,c='pink',label='AD mean')
# # plt.axhline(wm_cn_count_listmean,c='cyan',label='CN mean')
# # plt.title('AD vs CN PVS count (white matter)')
# # plt.text(1,np.max(wm_ad_count_list)-np.min(wm_cn_count_list)+2,str(bao))
# # plt.legend()
# # plt.show()
# # wm_fig_count.savefig(os.path.join(data_dir,'ADvsCN_pvscount_wm.jpeg'))

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

# # alldata_advol.to_csv(os.path.join(data_dir,'ADvolumes_wm.csv'), index=True)
# # alldata_cnvol.to_csv(os.path.join(data_dir,'CNvolumes_wm.csv'), index=True)
# # alldata_adcount.to_csv(os.path.join(data_dir,'ADcounts_wm.csv'), index=True)
# # alldata_cncount.to_csv(os.path.join(data_dir,'CNcounts_wm.csv'), index=True)





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
# # bg_fig_vol.savefig(os.path.join(data_dir,'ADvsCN_pvsvol_bg.jpeg'))

# # bg_fig_count = plt.figure(figsize=(10,5))
# # plt.scatter(ad_cases,bg_ad_count_list,c='red',label='AD')
# # plt.scatter(ad_cases,bg_cn_count_list,c='green',label='CN')
# # plt.axhline(bg_ad_count_listmean,c='pink',label='AD mean')
# # plt.axhline(bg_cn_count_listmean,c='cyan',label='CN mean')
# # plt.title('AD vs CN PVS count (basal ganglia)')
# # plt.text(1,np.max(bg_ad_count_list)-np.min(bg_cn_count_list)+2,str(bao))
# # plt.legend()
# # plt.show()
# # bg_fig_count.savefig(os.path.join(data_dir,'ADvsCN_pvscount_bg.jpeg'))

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

# # alldata_advol.to_csv(os.path.join(data_dir,'ADvolumes_bg.csv'), index=True)
# # alldata_cnvol.to_csv(os.path.join(data_dir,'CNvolumes_bg.csv'), index=True)
# # alldata_adcount.to_csv(os.path.join(data_dir,'ADcounts_bg.csv'), index=True)
# # alldata_cncount.to_csv(os.path.join(data_dir,'CNcounts_bg.csv'), index=True)

