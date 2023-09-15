import os
import numpy as np
import shutil
import pandas as pd
import sys

path_subj = '/Users/nanatang/Documents/GradResearch/SYT-MRIimages/Task1-selectfullseg'
files = os.listdir(path_subj)
subjects = [i for i in files if i.startswith('sub')]

t1_list = []
gt_list = []


# create a new folder in analysis folder for each subject, and gather all T1s into a list and all ground truths into a list
# now working only in new folder with all the copied data
for subject in subjects:
    subjdir = os.path.join(os.getcwd(),'frangi-valdo',subject+'-frangi')
    #subj_folder.append(subjdir)
    os.makedirs(subjdir,exist_ok=True)
    t1 = os.path.join(path_subj,subject,subject+'_space-T1_desc-masked_T1.nii.gz')
    shutil.copy(t1,subjdir)
    t1_list.append(os.path.join(subjdir,subject+'_space-T1_desc-masked_T1.nii.gz'))
    gt = os.path.join(path_subj,subject,subject+'_space-T1_desc-Rater1_PVSSeg.nii.gz')
    shutil.copy(gt,subjdir)
    gt_list.append(os.path.join(subjdir,subject+'_space-T1_desc-Rater1_PVSSeg.nii.gz'))


# skip preprocessing for now b/c takes too long; just segment the whole thing -- 6/15/23
# update: just do short preprocessing (bias field and denoise) -- 6/20/23
# update: include white + grey matter mask -- 6/21/23
# update: try brain extract, just white matter closed, and white matter closed and grey matter erode

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


t1_list_preproc = [] # list of preprocessed T1: brain extracted, white / grey matter extracted, ANTs preprocessed
mask_list = []



for subj_preproc in t1_list:

    #subject name
    subj_name = subj_preproc.split('/')[7][:7]
    subj_folder = subj_preproc.split('/'+subj_name+'_space',1)[0]


    ####----FSL brain extract----####
     
    brain_extract = os.path.join(subj_folder,subj_name+'-betbrain')
    os.system(f'{fsl_brain} \
                {subj_preproc} \
                {brain_extract} \
                -S \
                -f {0.2} \
                -g {0}' ) 


    
    #bias first, then denoise
    bias = os.path.join(subj_folder,subj_name+'-bias.nii.gz')
    biasdenoise = os.path.join(subj_folder,subj_name+'-bnpreproc.nii.gz')
    os.system(f'{bias_command} \
                -i {brain_extract} \
                -o {bias}')

    os.system(f'{denoise_command} \
                -i {bias} \
                -n Rician \
                -o {biasdenoise}')



 # -s {2} \
 #                -p {1} \
 #                -r {5} \
    ####----FSL segment----####
 
    os.system(f'{fsl_segment} \
                -t 1 \
                -n 3 \
                -H 0.1 \
                -I 4 \
                -l 20.0 \
                -o {biasdenoise[:-7]} \
                {biasdenoise[:-7]}') 
    
    ####---create grey + white matter mask---####

    # # grab grey and white matter outputs
    # grey_mask = os.path.join(subj_folder,subj_name+'-bnpreproc_pve_1.nii.gz')
    # white_mask = os.path.join(subj_folder,subj_name+'-bnpreproc_pve_2.nii.gz')
    # mask = os.path.join(subj_folder,subj_name+'-gmwmmask.nii.gz')


    # # combine 
    # os.system(f'{qit} MaskUnion \
    #           --left {white_mask} \
    #           --right {grey_mask} \
    #           --output {mask}')

    

    # ####---create white matter mask---####

    # white_mask = os.path.join(subj_folder,subj_name+'-bnpreproc_pve_2.nii.gz')
    # mask = os.path.join(subj_folder,subj_name+'-wmmaskclosed.nii.gz')

    # # close
    # os.system(f'{qit} MaskClose \
    #           --input {white_mask} \
    #           --num {3} \
    #           --output {mask}')



   ####---create white + deep grey matter - csf mask---####
   # steps:
        # 0. create deep grey matter mask
        # 1. combine white and deep grey matter masks and binarize
        # 2. subtract CSF mask from it
        # 3. fill holes that aren't grey matter
        # (?4 & 5?). subtract grey matter mask that has deep grey matter subtracted ?   


    # step 0
    deepgrey_folder = os.path.join(subj_folder,'deepgrey')  # this is: /Users/nanatang/.../frangi-valdo/subj/deepgrey
    deepgrey_out = os.path.join(deepgrey_folder,subj_name+'-deepgrey') # this is: /Users/nanatang/.../frangi-valdo/subj/deepgrey/deepgrey.nii.gz
    #os.makedirs(deepgrey_folder)
    #command = 'run_first_all'


    try:
        os.system(f'{fsl_deepgrey} \
                    -i {biasdenoise} \
                    -s L_Accu,L_Caud,L_Pall,L_Puta,R_Accu,R_Caud,R_Pall,R_Puta  \
                    -b \
                    -o {deepgrey_out}')
    except:
        pass

    deepgrey_mask = deepgrey_out+'_all_none_firstseg.nii.gz'



    # step 1
    white_mask = os.path.join(subj_folder,subj_name+'-bnpreproc_pve_2.nii.gz')
    wmdgm_mask = os.path.join(subj_folder,subj_name+'-wmdgmmask.nii.gz')

    # combine
    os.system(f'{qit} MaskUnion \
              --left {white_mask} \
              --right {deepgrey_mask} \
              --output {wmdgm_mask}')

    # binarize
    os.system(f'{qit} MaskBinarize \
              --input {wmdgm_mask} \
              --output {wmdgm_mask}')

    # step 2
    csf_mask = os.path.join(subj_folder,subj_name+'-bnpreproc_pve_0.nii.gz')
    subcsf_mask = os.path.join(subj_folder,subj_name+'-wmdgmmask-subcsf.nii.gz')

    os.system(f'{qit} MaskSet \
              --input {wmdgm_mask} \
              --mask {csf_mask} \
              --label {0} \
              --output {subcsf_mask}')

    # step 3
    # add: brainstem subtractor

    bs_folder = os.path.join(subj_folder,'brainstem')  # this is: /Users/nanatang/.../frangi-valdo/subj/deepgrey
    bs_out = os.path.join(bs_folder,subj_name) # this is: /Users/nanatang/.../frangi-valdo/subj/deepgrey/sub101-BrStem.nii.gz
    
    try:
        os.system(f'{fsl_deepgrey} \
                    -i {biasdenoise} \
                    -s BrStem  \
                    -b \
                    -o {bs_out}')
    except:
        pass
    
    bs_mask = bs_out+'-BrStem_first.nii.gz'
    nobs_mask = os.path.join(subj_folder,subj_name+'-wmdgmmask-subcsf-subbs.nii.gz')


    os.system(f'{qit} MaskSet \
                  --input {subcsf_mask} \
                  --mask {bs_mask} \
                  --label {0} \
                  --output {nobs_mask}')


    # step 4
    final_mask = os.path.join(subj_folder,subj_name+'-wmdgmmask-subcsf-subbs-closed.nii.gz')
    os.system(f'{qit} MaskClose \
              --input {nobs_mask} \
              --num {1} \
              --output {final_mask}')


    #sys.exit()


    # # step 3 without closing:
    # final mask = subcsf_mask


    ###### try above first #####

    # # step 4
    # gm_mask = os.path.join(subj_folder,subj_name+'-bnpreproc_pve_1.nii.gz')
    # grey_subdg = os.path.join(subj_folder,subj_name+'-deepgrey-subgrey.nii.gz')
    # os.system(f'{qit} MaskSet \
    #           --input {deepgrey_mask} \
    #           --mask {gm_mask} \
    #           --label {0} \
    #           --output {grey_subdg}')

    # # step 5
    # final_mask_ng = os.path.join(subj_folder,subj_name+'-wmdgmmask-subcsf-closed-ng.nii.gz')
    # os.system(f'{qit} MaskSet \
    #           --input {grey_subdg} \
    #           --mask {final_mask} \
    #           --label {0} \
    #           --output {final_mask_ng}')





    ############--------- for subjects with regional segmentation -----------##########

    # region_mask = os.path.join(subj_folder,subj_name+'_space-T1_desc-masked_Regions.nii.gz')
    # final_mask = os.path.join(subj_folder,subj_name+'-region-mask.nii.gz')
    # # binarize
    # os.system(f'{qit} MaskBinarize \
    #           --input {region_mask} \
    #           --output {final_mask}')

    ############--------- for subjects with regional segmentation -----------##########

    mask_list.append(final_mask)
    # mask_list.append(final_mask_ng)


    print('preproc done for '+subj_name+'!')



    ##### Breaking analysis up into 3 parts: just white matter, just grey matter, and both

    ############# both first:
    # hessian calculation
    hes =  os.path.join(subj_folder, subj_name+'-hessian.nii.gz')
    os.system(f'{qit} VolumeFilterHessian \
              --input {biasdenoise} \
              --mask {final_mask} \
              --mode Norm \
              --output {hes}')

    hes_stats = os.path.join(subj_folder, subj_name+'-hessianstats.csv')
    os.system(f'{qit} VolumeMeasure \
              --input {hes} \
              --output {hes_stats}')

    hes_csv = pd.read_csv(hes_stats,index_col=0)
    half_max = hes_csv.loc['max'][0]/2


    # frangi calculation
    frangi_mask = os.path.join(subj_folder, subj_name+'-frangimask.nii.gz')
    os.system(f'{qit} VolumeFilterFrangi \
              --input {biasdenoise} \
              --mask {final_mask} \
              --low {0.1} \
              --high {5.0} \
              --scales {10} \
              --gamma {half_max} \
              --dark \
              --output {frangi_mask}')

    # threshold calculation
    # for now, set threshold to optimal threshold you found with sub-101 with bias / denoise 
    t = 0.0025

    frangi_thresholded = os.path.join(subj_folder, subj_name+'-frangimask-thresholded.nii.gz')
    os.system(f'{qit} VolumeThreshold \
              --input {frangi_mask} \
              --threshold {t} \
              --output {frangi_thresholded}')

    # next step: create some kind of algorithm to optimize threshold per subject



    # dice score
    dice_subj = os.path.join(subj_folder, subj_name+'-dicescore.csv')

    os.system(f'{qit} MaskDiceBatch \
              --left {frangi_thresholded} \
              --right {gt} \
              --output {dice_subj}')


    ############## white matter:
    # hessian calculation
    hes_wm =  os.path.join(subj_folder, subj_name+'-hessian-wm.nii.gz')
    os.system(f'{qit} VolumeFilterHessian \
              --input {biasdenoise} \
              --mask {white_mask} \
              --mode Norm \
              --output {hes_wm}')

    hes_stats_wm= os.path.join(subj_folder, subj_name+'-hessianstats-wm.csv')
    os.system(f'{qit} VolumeMeasure \
              --input {hes_wm} \
              --output {hes_stats_wm}')

    hes_csv_wm = pd.read_csv(hes_stats_wm,index_col=0)
    half_max = hes_csv_wm.loc['max'][0]/2


    # frangi calculation
    frangi_mask_wm= os.path.join(subj_folder, subj_name+'-frangimask-wm.nii.gz')
    os.system(f'{qit} VolumeFilterFrangi \
              --input {biasdenoise} \
              --mask {white_mask} \
              --low {0.1} \
              --high {5.0} \
              --scales {10} \
              --gamma {half_max} \
              --dark \
              --output {frangi_mask_wm}')    


        # threshold calculation
    # for now, set threshold to optimal threshold you found with sub-101 with bias / denoise 
    t = 0.0009

    frangi_thresholded_wm = os.path.join(subj_folder, subj_name+'-frangimask-thresholded-wm.nii.gz')
    os.system(f'{qit} VolumeThreshold \
              --input {frangi_mask_wm} \
              --threshold {t} \
              --output {frangi_thresholded_wm}')

    # next step: create some kind of algorithm to optimize threshold per subject



    # dice score
    dice_subj_wm = os.path.join(subj_folder, subj_name+'-dicescore-wm.csv')

    os.system(f'{qit} MaskDiceBatch \
              --left {frangi_thresholded_wm} \
              --right {gt} \
              --output {dice_subj_wm}')

    ############## deep grey:

    hes_gm =  os.path.join(subj_folder, subj_name+'-hessian-gm.nii.gz')
    os.system(f'{qit} VolumeFilterHessian \
              --input {biasdenoise} \
              --mask {deepgrey_mask} \
              --mode Norm \
              --output {hes_gm}')

    hes_stats_gm= os.path.join(subj_folder, subj_name+'-hessianstats-gm.csv')
    os.system(f'{qit} VolumeMeasure \
              --input {hes_gm} \
              --output {hes_stats_gm}')

    hes_csv_gm = pd.read_csv(hes_stats_gm,index_col=0)
    half_max = hes_csv_gm.loc['max'][0]/2


    # frangi calculation
    frangi_mask_gm= os.path.join(subj_folder, subj_name+'-frangimask-gm.nii.gz')
    os.system(f'{qit} VolumeFilterFrangi \
              --input {biasdenoise} \
              --mask {deepgrey_mask} \
              --low {0.1} \
              --high {5.0} \
              --scales {10} \
              --gamma {half_max} \
              --dark \
              --output {frangi_mask_gm}')    


        # threshold calculation
    # for now, set threshold to optimal threshold you found with sub-101 with bias / denoise 
    t = 0.0026

    frangi_thresholded_gm = os.path.join(subj_folder, subj_name+'-frangimask-thresholded-gm.nii.gz')
    os.system(f'{qit} VolumeThreshold \
              --input {frangi_mask_gm} \
              --threshold {t} \
              --output {frangi_thresholded_gm}')

    # next step: create some kind of algorithm to optimize threshold per subject



    # dice score
    dice_subj_gm = os.path.join(subj_folder, subj_name+'-dicescore-gm.csv')

    os.system(f'{qit} MaskDiceBatch \
              --left {frangi_thresholded_gm} \
              --right {gt} \
              --output {dice_subj_gm}')

    ############# insert IQR scaling




    d = pd.read_csv(dice_subj,index_col=0)
    dice = d.loc['region1'][0]

    d = pd.read_csv(dice_subj_wm,index_col=0)
    dice_wm = d.loc['region1'][0]

    d = pd.read_csv(dice_subj_gm,index_col=0)
    dice_gm = d.loc['region1'][0]

    alldice = [dice,dice_wm,dice_gm]

    subname.append(subj_name)
    dicescore.append(alldice)

    print('subject '+ subj_name + ' done!')

    print(dice)

    #sys.exit()




dicecsv = pd.DataFrame(data=dicescore,index = subname,columns =['total dice','wm dice','gm dice']).rename_axis('subjects')
dicecsv.loc['avg'] = [np.mean(dicescore[:,0]),np.mean(dicescore[:,1]),np.mean(dicescore[:,2])]

dicecsv.to_csv('/Users/nanatang/Documents/GradResearch/frangi-valdo/dicescore_all.csv', index=True)







