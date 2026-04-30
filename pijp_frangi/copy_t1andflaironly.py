import os
import shutil
import Path

parent = '/m/Researchers/SerenaT/deeppvs/for_nnunet/gt_mcpvs_preprocessed3'
destination = '/m/Researchers/SerenaT/deeppvs/for_nnunet/gt_mcpvs_preprocessed3_t1flaironly'

folder_list = [os.path.join(parent,s) for s in os.listdir(parent) if not s.startswith('.')]
subjname_list = [s for s in os.listdir(parent) if not s.startswith('.')]

for subj_dir,name in (folder_list,subjname_list):
    output_dir = os.path.join(destination,subj_dir)
    os.makedirs(output_dir,exist_ok=True)    # in case it doesn't exist
    # files I need: t1, talairach, raw flair, wmmask
    #t1 = os.path.join(subj_dir, subject + '.T1.nii.gz')
    t1 = [os.path.join(subj_dir,t1) for t1 in os.listdir(subj_dir) if t1.endswith('-T1bcbrainmask_norm.nii.gz')][0]
    #subjname = Path(t1).stem.replace('.T1.nii', '')   # redefine subject based on full image name
    shutil.copy(t1,os.path.join(output_dir))
    
    flair = [os.path.join(subj_dir,flair) for flair in os.listdir(subj_dir) if flair.endswith('-FLAIRbcreg.nii.gz')][0]
    shutil.copy(flair,os.path.join(output_dir))
   
