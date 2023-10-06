import os
import numpy as np
import shutil
import pandas as pd
import sys
import gzip
import nibabel as nib
from matplotlib import pyplot as plt
from scipy import stats



path_subj = '/Users/nanatang/Documents/GradResearch/ADNI3_samples_fslseg'
files = os.listdir(path_subj)
subjects = [i for i in files if i.startswith('sub')]
#print(subjects)
t1_list = []
#flair_list = []
#gt_list = []

#sys.exit()
##### stopping point: redo the directories and subfolders
######### - already commented out all of the flair stuff, also included the correct mask measure code

# create a new folder in analysis folder for each subject, and gather all T1s into a list and all ground truths into a list
# now working only in new folder with all the copied data
for subject in subjects:
    # subjdir = os.path.join(os.getcwd(),'frangi-valdo',subject+'-frangi')    # /Users/nanatang/Documents/GradResearch/frangi-valdo/sub-101-frangi
    # os.makedirs(subjdir,exist_ok=True)
    t1 = os.path.join(path_subj,subject,subject+'-T1.nii.gz')  # '/Users/nanatang/Documents/GradResearch/SYT-MRIimages/Task1-selectfullseg/sub-101/sub-101-..T1.nii.ga'


    # rotate the image bc somehow that works ??
    img = nib.load(t1)
    data = img.get_fdata()
    data_rot = np.rot90(data,axes=(2,1))
    img_rot = nib.Nifti1Image(data_rot,np.eye(4))
    nib.save(img_rot,os.path.join(path_subj,subject,subject+'-T1rot.nii.gz'))  
    t1_rot = os.path.join(path_subj,subject,subject+'-T1rot.nii.gz')

    #shutil.copy(t1,subjdir)
    t1_list.append(t1_rot)
    # gt = os.path.join(path_subj,subject,subject+'_space-T1_desc-Rater1_PVSSeg.nii.gz')
    # shutil.copy(gt,subjdir)
    # gt_list.append(os.path.join(subjdir,subject+'_space-T1_desc-Rater1_PVSSeg.nii.gz'))

    # #new addition: wmh > put flair into new folder called lesion
    # os.makedirs(os.path.join(subjdir,'lesion'),exist_ok=True)
    # new_flair_name = os.path.join(subjdir,'lesion',subject+'-FLAIR.nii.gz')
    # flair = os.path.join(path_subj,subject,subject+'_space-T1_desc-masked_FLAIR.nii.gz')
    # shutil.copy(flair,os.path.join(subjdir,'lesion'))
    # os.rename(os.path.join(subjdir,'lesion',subject+'_space-T1_desc-masked_FLAIR.nii.gz'),new_flair_name)
    # unzipped = os.path.join(subjdir,'lesion',subject+'-FLAIR.nii')
    # with gzip.open(new_flair_name, 'rb') as f_in:
    #     with open(unzipped, 'wb') as f_out:
    #         shutil.copyfileobj(f_in, f_out)

    # flair_list.append(unzipped)



# # skip preprocessing for now b/c takes too long; just segment the whole thing -- 6/15/23
# # update: just do short preprocessing (bias field and denoise) -- 6/20/23
# # update: include white + grey matter mask -- 6/21/23
# # update: try brain extract, just white matter closed, and white matter closed and grey matter erode
# # update: specific BG structures only, subtract brainstem
# # update: subtract brain stem
# # update: trying denoise with QIT —> decide whether or not to keep
# # update: include white matter lesion removal

#print(t1_list)
#sys.exit()
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
#wmh_list = [] # list of wmh masks
mask_list = []

# for subj_preproc,subj_name in zip(t1_list,subjects):

#     #print("subj name: "+subj_name)
#     #print("subj preproc: "+subj_preproc)
#     #sys.exit()
#     #subject name
#     #subj_name = subj_preproc.split('/')[7][:7]      # in this case, subject name is the subject folder name
#     #subj_folder = subj_preproc.split('/'+subj_name+'-ad',1)[0]
#     subj_folder = os.path.join(path_subj,subj_name)
#     #print("subj folder: "+subj_folder)
#     #sys.exit()

#      ####----FSL brain extract----####
     
#     brain_extract = os.path.join(subj_folder,subj_name+'-betbrain.nii.gz')

#     if '-5-' in subj_name:
#         os.system(f'{fsl_brain} \
#                     {subj_preproc} \
#                     {brain_extract} \
#                     -R \
#                     -f {0.5} \
#                     -g {-0.2}' ) 
#     elif '-8-' in subj_name:
#         os.system(f'{fsl_brain} \
#                     {subj_preproc} \
#                     {brain_extract} \
#                     -R \
#                     -f {0.6} \
#                     -g {0}' )
#     elif '-10-' in subj_name:
#         os.system(f'{fsl_brain} \
#                     {subj_preproc} \
#                     {brain_extract} \
#                     -R \
#                     -f {0.6} \
#                     -g {0}' )   
#     else:
#          os.system(f'{fsl_brain} \
#                     {subj_preproc} \
#                     {brain_extract} \
#                     -R \
#                     -f {0.5} \
#                     -g {0}' )                       

#     print('brain extraction done!')

#     ####----ANTs preproc---#### ------> not doing it for ADNI since it should already be preprocessed

#     # trying no preproc
#     # trying bias only
#     # back to denoising - try default again, but with shrink factor

#     #biasdenoise = brain_extract

#     #bias first, then denoise
#     # bias = os.path.join(subj_name,subj_name+'-bias.nii.gz')    
#     # os.system(f'{bias_command} \
#     #             -i {brain_extract} \
#     #             -o {bias}')


#     # os.system(f'{denoise_command} \
#     #             -i {bias} \
#     #             -s {2} \
#     #             -p {1} \
#     #             -r {13} \
#     #             -n Rician \
#     #             -o {biasdenoise}')
#     #biasdenoise = bias

# #               -s 2 \
# #               -p 1 \
# #               -r 13 \
#     #t1_list_preproc.append(biasdenoise)


#     #print('ANTs processing done!')
#     ####----FSL segment----####
 
#     os.system(f'{fsl_segment} \
#                 -t 1 \
#                 -n 3 \
#                 -H 0.1 \
#                 -I 4 \
#                 -l 20.0 \
#                 -o {brain_extract[:-7]} \
#                 {brain_extract[:-7]}') 


#     print('FSL segmentation done!')
#     ####---create grey + white matter mask---####

#     # # grab grey and white matter outputs
#     # grey_mask = os.path.join(subj_folder,subj_name+'-bnpreproc_pve_1.nii.gz')
#     # white_mask = os.path.join(subj_folder,subj_name+'-bnpreproc_pve_2.nii.gz')
#     # mask = os.path.join(subj_folder,subj_name+'-gmwmmask.nii.gz')


#     # # combine 
#     # os.system(f'{qit} MaskUnion \
#     #           --left {white_mask} \
#     #           --right {grey_mask} \
#     #           --output {mask}')

    

#     # ####---create white matter mask---####

#     # white_mask = os.path.join(subj_folder,subj_name+'-bnpreproc_pve_2.nii.gz')
#     # mask = os.path.join(subj_folder,subj_name+'-wmmaskclosed.nii.gz')

#     # # close
#     # os.system(f'{qit} MaskClose \
#     #           --input {white_mask} \
#     #           --num {3} \
#     #           --output {mask}')



#    ####---CURRENT: create white + deep grey matter - csf mask---####
#    # steps:
#         # 0. create deep grey matter mask, create brainstem mask, create cerebellum mask
#         # 1. combine white and deep grey matter masks and binarize
#         # 2. subtract CSF mask from it
#         # 3. combine brainstem / cerebellum
#         # 4. subtract brainstem / cerebellum
#         # 5 (optional). fill holes that aren't grey matter
#         # (?6 & 7?). subtract grey matter mask that has deep grey matter subtracted ?   


#     # step 0
#     deepgrey_folder = os.path.join(subj_folder,'deepgrey')  # this is: /Users/nanatang/.../frangi-valdo/subj/deepgrey
#     deepgrey_out = os.path.join(deepgrey_folder,subj_name+'-deepgrey') # this is: /Users/nanatang/.../frangi-valdo/subj/deepgrey/deepgrey.nii.gz
#     #os.makedirs(deepgrey_folder)
#     #command = 'run_first_all'
#     try:
#         os.system(f'{fsl_deepgrey} \
#                     -i {brain_extract} \
#                     -s L_Thal,L_Accu,L_Caud,L_Pall,L_Puta,R_Accu,R_Caud,R_Pall,R_Puta,R_Thal  \
#                     -b \
#                     -o {deepgrey_out}')
#     except:
#         pass

#     deepgrey_mask = deepgrey_out+'_all_none_firstseg.nii.gz'

#     # binarize
#     os.system(f'{qit} MaskBinarize \
#               --input {deepgrey_mask} \
#               --output {deepgrey_mask}')


#     print('deepgrey mask is done: '+deepgrey_mask)


#     bs_folder = os.path.join(subj_folder,'brainstem')  # this is: /Users/nanatang/.../frangi-valdo/subj/brainstem
#     bs_out = os.path.join(bs_folder,subj_name) # this is: /Users/nanatang/.../frangi-valdo/subj/brainstem/sub101-BrStem.nii.gz
    
#     try:
#         os.system(f'{fsl_deepgrey} \
#                     -i {brain_extract} \
#                     -s BrStem  \
#                     -b \
#                     -o {bs_out}')
#     except:
#         pass
    
#     bs_mask = bs_out+'-BrStem_first.nii.gz'

#     # binarize
#     os.system(f'{qit} MaskBinarize \
#               --input {bs_mask} \
#               --output {bs_mask}')

#     print('brainstem mask is done: '+bs_mask)


#     cb_folder = os.path.join(subj_folder,'cerebellum')  # this is: /Users/nanatang/.../frangi-valdo/subj/cerebellum    
#     os.makedirs(cb_folder,exist_ok=True)
#     shutil.copy(brain_extract,cb_folder)

#     baseim_cereb = os.path.join(cb_folder,subj_name+'-betbrain.nii.gz')
#     txfm_out = os.path.join(cb_folder,subj_name+'-std_sub')
#     rcb_out = os.path.join(cb_folder,subj_name+'-rcerebellum.nii.gz') # this is: /Users/nanatang/.../frangi-valdo/subj/cerebellum/sub101-BrStem.nii.gz
#     lcb_out = os.path.join(cb_folder,subj_name+'-lcerebellum.nii.gz') # this is: /Users/nanatang/.../frangi-valdo/subj/cerebellum/sub101-BrStem.nii.gz

#     fsl_model = '/Users/nanatang/Documents/GradResearch/fsl/data/first/models_336_bin'

#     try:
#         os.system(f'{fsl_firstflirt} \
#                     {baseim_cereb} \
#                     {txfm_out} \
#                     -b \
#                     -cort')

#         print('first flirt is done!')


#         cort = txfm_out+'_cort.mat'
#         m_right = os.path.join(fsl_model,'intref_puta','R_Cereb.bmv')
#         intref_right = os.path.join(fsl_model,'05mm','R_Puta_05mm.bmv')
#         m_left = os.path.join(fsl_model,'intref_puta','L_Cereb.bmv')
#         intref_left = os.path.join(fsl_model,'05mm','L_Puta_05mm.bmv')

#         print('baseim_cereb '+baseim_cereb)

#         print('cort: '+cort)
#         print('m_right: '+m_right)
#         print('intref_right: '+intref_right)
#         print('m_left: '+m_left)
#         print('intref_left: '+intref_left)


#         os.system(f'{fsl_runfirst} \
#                     -i {baseim_cereb} \
#                     -t {cort} \
#                     -n {40} \
#                     -o {rcb_out} \
#                     -m {m_right} \
#                     -intref {intref_right}')
        
#         os.system(f'{fsl_runfirst} \
#                     -i {baseim_cereb} \
#                     -t {cort} \
#                     -n {40} \
#                     -o {lcb_out} \
#                     -m {m_left}\
#                     -intref {intref_left}') 
#     except:
#         pass
    

#     # binarize
#     cb_mask = os.path.join(cb_folder,subj_name+'-cerebellum.nii.gz')
#     os.system(f'{qit} MaskBinarize \
#               --input {lcb_out} \
#               --output {lcb_out}')  
#     os.system(f'{qit} MaskBinarize \
#               --input {rcb_out} \
#               --output {rcb_out}')  
#     os.system(f'{qit} MaskUnion \
#               --left {rcb_out} \
#               --right {lcb_out} \
#               --output {cb_mask}')


#     print('cerebellum mask is done: '+cb_mask)

    
#     # step 1
#     white_mask = os.path.join(subj_folder,subj_name+'-betbrain_pve_2.nii.gz')
    
#     os.system(f'{qit} MaskBinarize \
#               --input {white_mask} \
#               --output {white_mask}')  


#     wmdgm_mask = os.path.join(subj_folder,subj_name+'-wmdgmmask.nii.gz')

#     # combine
#     os.system(f'{qit} MaskUnion \
#               --left {white_mask} \
#               --right {deepgrey_mask} \
#               --output {wmdgm_mask}')

#     # binarize
#     os.system(f'{qit} MaskBinarize \
#               --input {wmdgm_mask} \
#               --output {wmdgm_mask}')

#     # step 2
#     csf_mask = os.path.join(subj_folder,subj_name+'-betbrain_pve_0.nii.gz')

#     os.system(f'{qit} MaskBinarize \
#               --input {csf_mask} \
#               --output {csf_mask}')  


#     subcsf_mask = os.path.join(subj_folder,subj_name+'-wmdgmmask-subcsf.nii.gz')

#     os.system(f'{qit} MaskSet \
#               --input {wmdgm_mask} \
#               --mask {csf_mask} \
#               --label {0} \
#               --output {subcsf_mask}')

#     # step 3
#     nobs_mask = os.path.join(subj_folder,subj_name+'-wmdgmmask-subcsf-subbscb.nii.gz')
#     bscb_mask = os.path.join(subj_folder,subj_name+'-brstm-cereb-mask.nii.gz')

#     # combine
#     os.system(f'{qit} MaskUnion \
#               --left {bs_mask} \
#               --right {cb_mask} \
#               --output {bscb_mask}')


#     os.system(f'{qit} MaskSet \
#                   --input {subcsf_mask} \
#                   --mask {bscb_mask} \
#                   --label {0} \
#                   --output {nobs_mask}')

#     #final_mask = nobs_mask

#     # step 4
#     final_mask = os.path.join(subj_folder,subj_name+'-allmask-closed.nii.gz')
#     os.system(f'{qit} MaskClose \
#               --input {nobs_mask} \
#               --num {1} \
#               --output {final_mask}')


#     # # step 3 without closing:
#     # final mask = subcsf_mask


#     ###### try above first #####

#     # # step 4
#     # gm_mask = os.path.join(subj_folder,subj_name+'-bnpreproc_pve_1.nii.gz')
#     # grey_subdg = os.path.join(subj_folder,subj_name+'-deepgrey-subgrey.nii.gz')
#     # os.system(f'{qit} MaskSet \
#     #           --input {deepgrey_mask} \
#     #           --mask {gm_mask} \
#     #           --label {0} \
#     #           --output {grey_subdg}')

#     # # step 5
#     # final_mask_ng = os.path.join(subj_folder,subj_name+'-wmdgmmask-subcsf-closed-ng.nii.gz')
#     # os.system(f'{qit} MaskSet \
#     #           --input {grey_subdg} \
#     #           --mask {final_mask} \
#     #           --label {0} \
#     #           --output {final_mask_ng}')





#     ############--------- for subjects with regional segmentation -----------##########

#     # region_mask = os.path.join(subj_folder,subj_name+'_space-T1_desc-masked_Regions.nii.gz')
#     # final_mask = os.path.join(subj_folder,subj_name+'-region-mask.nii.gz')
#     # # binarize
#     # os.system(f'{qit} MaskBinarize \
#     #           --input {region_mask} \
#     #           --output {final_mask}')

#     ############--------- for subjects with regional segmentation -----------##########

#     mask_list.append(final_mask)
#     print('mask done!')

#     # biasdenoise = os.path.join(subj_name,subj_name+'-bnpreproc.nii.gz')
#     # bias_nan = os.path.join(subj_name,subj_name+'-bias-nan.nii.gz')

#     # os.system(f'{fsl_maths}  \
#     #             {bias} \
#     #             -nan \
#     #             {bias_nan}')

#     # os.system(f'{qit} VolumeFilterNLM \
#     #             --input {bias_nan} \
#     #             --rician \
#     #             --output {biasdenoise}')

#     # print('denoise done!')
#     t1_list_preproc.append(brain_extract)

#     # mask_list.append(final_mask_ng)



#     # ##########--------------- WMH calculation with LST -----------------#############

#     # # working in lesion folder

#     # flair = '\'/Users/nanatang/Documents/GradResearch/frangi-valdo/'+subj_name+'-frangi/lesion/'+subj_name+'-FLAIR.nii\''

#     # unzipped = os.path.join(subj_folder,subj_name+'-bnpreproc.nii')

#     # with gzip.open(biasdenoise, 'rb') as f_in:
#     #     with open(unzipped, 'wb') as f_out:
#     #         shutil.copyfileobj(f_in, f_out)

#     # shutil.copy(unzipped,os.path.join(subj_folder,'lesion'))
#     # t1 = '\'/Users/nanatang/Documents/GradResearch/frangi-valdo/'+subj_name+'-frangi/lesion/'+subj_name+'-bnpreproc.nii\''

#     # wmh_remove = f"\"addpath('/Users/nanatang/Documents/GradResearch/spm12');spm_jobman('initcfg'); \
#     #             ps_LST_lpa({flair},{t1},'yes'); \
#     #             exit\""
    
#     # os.system(f'{matlab} \
#     #             -nodesktop \
#     #             -noFigureWindows \
#     #             -nosplash \
#     #             -r \
#     #             {wmh_remove}')

#     # wmh_list.append(os.path.join(subj_folder,'lesion','ples_lpa_mr'+subj_name+'-FLAIR.nii'))

#     # #########--------------- WMH calculation with LST -----------------#############




#     print('preproc done for '+subj_name+'!')

    #sys.exit()


#print(t1_list_preproc)
#print(gt_list)

#sys.exit()

#dicescore = []


#print('subj list: '+subjects)
t1_list_preproc = ['/Users/nanatang/Documents/GradResearch/ADNI3_samples/sub-4-ad/sub-4-ad-betbrain.nii.gz',\
                    '/Users/nanatang/Documents/GradResearch/ADNI3_samples/sub-1-ad/sub-1-ad-betbrain.nii.gz', \
                    '/Users/nanatang/Documents/GradResearch/ADNI3_samples/sub-5-ad/sub-5-ad-betbrain.nii.gz',\
                    '/Users/nanatang/Documents/GradResearch/ADNI3_samples/sub-10-cn/sub-10-cn-betbrain.nii.gz',\
                    '/Users/nanatang/Documents/GradResearch/ADNI3_samples/sub-11-cn/sub-11-cn-betbrain.nii.gz',\
                    '/Users/nanatang/Documents/GradResearch/ADNI3_samples/sub-7-cn/sub-7-cn-betbrain.nii.gz',\
                    '/Users/nanatang/Documents/GradResearch/ADNI3_samples/sub-12-cn/sub-12-cn-betbrain.nii.gz', \
                    '/Users/nanatang/Documents/GradResearch/ADNI3_samples/sub-2-ad/sub-2-ad-betbrain.nii.gz',\
                    '/Users/nanatang/Documents/GradResearch/ADNI3_samples/sub-6-ad/sub-6-ad-betbrain.nii.gz',\
                    '/Users/nanatang/Documents/GradResearch/ADNI3_samples/sub-9-cn/sub-9-cn-betbrain.nii.gz',\
                    '/Users/nanatang/Documents/GradResearch/ADNI3_samples/sub-8-cn/sub-8-cn-betbrain.nii.gz', \
                    '/Users/nanatang/Documents/GradResearch/ADNI3_samples/sub-3-ad/sub-3-ad-betbrain.nii.gz']

mask_list = ['/Users/nanatang/Documents/GradResearch/ADNI3_samples/sub-4-ad/sub-4-ad-allmask-closed.nii.gz',\
                    '/Users/nanatang/Documents/GradResearch/ADNI3_samples/sub-1-ad/sub-1-ad-allmask-closed.nii.gz', \
                    '/Users/nanatang/Documents/GradResearch/ADNI3_samples/sub-5-ad/sub-5-ad-allmask-closed.nii.gz',\
                    '/Users/nanatang/Documents/GradResearch/ADNI3_samples/sub-10-cn/sub-10-cn-allmask-closed.nii.gz',\
                    '/Users/nanatang/Documents/GradResearch/ADNI3_samples/sub-11-cn/sub-11-cn-allmask-closed.nii.gz',\
                    '/Users/nanatang/Documents/GradResearch/ADNI3_samples/sub-7-cn/sub-7-cn-allmask-closed.nii.gz', \
                    '/Users/nanatang/Documents/GradResearch/ADNI3_samples/sub-12-cn/sub-12-cn-allmask-closed.nii.gz', \
                    '/Users/nanatang/Documents/GradResearch/ADNI3_samples/sub-2-ad/sub-2-ad-allmask-closed.nii.gz',\
                    '/Users/nanatang/Documents/GradResearch/ADNI3_samples/sub-6-ad/sub-6-ad-allmask-closed.nii.gz',\
                    '/Users/nanatang/Documents/GradResearch/ADNI3_samples/sub-9-cn/sub-9-cn-allmask-closed.nii.gz',\
                    '/Users/nanatang/Documents/GradResearch/ADNI3_samples/sub-8-cn/sub-8-cn-allmask-closed.nii.gz',\
                    '/Users/nanatang/Documents/GradResearch/ADNI3_samples/sub-3-ad/sub-3-ad-allmask-closed.nii.gz']

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

for subj,m,subj_name in zip(t1_list_preproc,mask_list,subjects):

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


ad_cases = [1,2,3,4,5,6]
cn_cases = [7,8,9,10,11,12]

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

ad_cases.append(ad_casesmean)
cn_cases.append(cn_casesmean)
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
