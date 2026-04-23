import os
import subprocess
import numpy as np
import nibabel as nib
import shutil
import pandas as pd
import argparse
from nipype.interfaces.spm import NewSegment
from nipype.interfaces.spm.base import Info
import gzip
from pathlib import Path

##########################################
###     some useful functions       ####
##########################################

def run_command(cmd_list):
    """Run a command and raise error if it fails"""
    subprocess.run(cmd_list, check=True)


def spm12_brain_extract(t1_path, spm12_dir, output_mask, output_brain):    
    # Configure SPM12 tissue probability maps
    tpm = spm12_dir + '/tpm/TPM.nii'
    tissue_list = []
    for i, (ngaus, native) in enumerate([
        (1, (1, 0)),   # GM
        (1, (1, 0)),   # WM
        (2, (1, 0)),   # CSF
        (3, (0, 0)),   # skull   — excluded
        (4, (0, 0)),   # soft tissue — excluded
        (2, (0, 0)),   # air/background — excluded
    ], start=1):
        tissue_list.append(((tpm, i), ngaus, native, (0, 0)))

    # Run segmentation
    seg = NewSegment()
    seg.inputs.channel_files = t1_path
    seg.inputs.channel_info = (0, np.inf, (False, False))  # skip bias correction
    seg.inputs.tissues = tissue_list
    seg.inputs.write_deformation_fields = [False, False]
    result = seg.run()

    # Build brain mask from GM + WM + CSF probability maps
    prob_maps = result.outputs.native_class_images[:3]  # classes 1-3 only
    ref = nib.load(t1_path)
    brain_prob = sum(nib.load(p[0]).get_fdata() for p in prob_maps)
    mask = (brain_prob > 0.5).astype(np.int16)
    stripped = ref.get_fdata() * mask

    # Save outputs
    # stem = Path(t1_path).stem.replace('.nii', '')
    mask_path = output_mask
    brain_path = output_brain
    nib.save(nib.Nifti1Image(mask, ref.affine, ref.header), mask_path)
    nib.save(nib.Nifti1Image(stripped, ref.affine, ref.header), brain_path)

    return brain_path


def gunzip(path):
    out_path = path.replace('.nii.gz', '.nii')
    with gzip.open(path, 'rb') as f_in, open(out_path, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    return out_path
##########################################
###     running preprocessing       ####
##########################################
os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'

# parent_dir = '/m/Researchers/SerenaT/deeppvs/for_nnunet/ADNI3_preprocessed'
# dx_names = ['EMCI','AD','MCI','CN','LMCI','SMC']

# failed_subjects = []  # Add this before your loop

def main():
    # Create the parser
    parser = argparse.ArgumentParser(description='Preprocess images for a subject')

    # # You can add more arguments as needed
    # parser.add_argument('--output', type=str, default='./output',
    #                     help='Output directory (default: ./output)')
    # parser.add_argument('--format', type=str, default='png',
    #                     help='Output image format (default: png)')
    
    # Add arguments
    parser.add_argument('--subj_dir', type=str, 
                        help='Path to the output folder')
    parser.add_argument('--subject', type=str, 
                        help='Subject name')
    parser.add_argument('--output_folder', type=str, 
                        help='Path to the output folder')
    
    
    # Parse the arguments
    args = parser.parse_args()
    
    output_dir = args.output_folder
    subj_dir = args.subj_dir
    subject = args.subject
    
    print(f"Processing images in: {subj_dir}")
    #print(f"Output directory: {output_dir}")
    failed_subjects = []  # Add this before your loop
    spm12_path = '/opt/mathworks/MatlabToolkits/spm12_r7219'

    try:
        #subject = subj_dir.split('/')[-1]
        os.makedirs(output_dir,exist_ok=True)    # in case it doesn't exist
        # files I need: t1, talairach, raw flair, wmmask
        #t1 = os.path.join(subj_dir, subject + '.T1.nii.gz')
        t1 = [os.path.join(subj_dir,t1) for t1 in os.listdir(subj_dir) if t1.endswith('.T1.nii.gz')][0]
        subjname = Path(t1).stem.replace('.nii', '')   # redefine subject based on full image name
        shutil.copy(t1,os.path.join(output_dir,subjname + '-T1raw.nii.gz'))
        t1 = os.path.join(output_dir, subjname + '-T1raw.nii.gz')

        flair = [os.path.join(subj_dir,flair) for flair in os.listdir(subj_dir) if flair.endswith('.FLAIR.nii.gz')][0]
        shutil.copy(flair,os.path.join(output_dir,subjname + '-FLAIRraw.nii.gz'))
        flair = os.path.join(output_dir, subjname + '-FLAIRraw.nii.gz')
       
        
        # Check T1 exists
        if not os.path.exists(t1):
            raise FileNotFoundError(f"T1 not found: {t1}")
        
        # Check flair exists
        if not os.path.exists(flair):
            raise FileNotFoundError(f"flair not found: {flair}")
        
       
    

        ##########################################
        #####           processing            ####
        ##########################################
        
        # N4 bias field correction for T1
        t1_bc = os.path.join(output_dir,subjname+'-T1bc.nii.gz')
        run_command(['N4BiasFieldCorrection', '-i', t1, '-o', t1_bc])

        ## identify raw flair and processing with N4 bias field correction
        # flair_bc = os.path.join(subj_dir,subject+'-FLAIRbc.nii.gz')
        flair_bc = os.path.join(output_dir,subjname+'-FLAIRbc.nii.gz')
        run_command(['N4BiasFieldCorrection', '-i', flair, '-o', flair_bc])


        # #### register flair to t1 and to template:
        flair_bcreg = os.path.join(output_dir,subjname+'-FLAIRbcreg.nii.gz')
        run_command(['flirt', '-in', flair_bc, '-ref',t1_bc,'-out',flair_bcreg, '-dof', '6'])

        ### brain extraction using SPM
        t1_bc_brainextract = os.path.join(output_dir,subjname+'-T1bcbrainmask.nii.gz')
        brain_mask = os.path.join(output_dir,subjname+'-brainmask.nii.gz')
        unzipped_t1 = gunzip(t1_bc)
        spm12_brain_extract(unzipped_t1,spm12_path,brain_mask,t1_bc_brainextract)

        ## intensity normalization with fuzzy-C means: https://github.com/jcreinhold/intensity-normalization?tab=readme-ov-file
        t1_bc_brainextract_norm = os.path.join(output_dir,subjname+'-T1bcbrainmask_norm.nii.gz')
        run_command(['intensity-normalize', 'zscore', t1_bc_brainextract, '-o', t1_bc_brainextract_norm])


        #break  # only do one for testing
    except FileNotFoundError as e:
        print(f"SKIPPING {subject}: {str(e)}")
        failed_subjects.append({'subject': subject, 'reason': str(e)})
        
    except Exception as e:
        print(f"ERROR processing {subject}: {str(e)}")
        failed_subjects.append({'subject': subject, 'reason': f'Processing error: {str(e)}'})
    
    #     break   # just do 1 subject first
    # break   

    # Save failed subjects at the end
    if failed_subjects:
        with open('./failed_subjects.txt', 'w') as f:
            for failure in failed_subjects:
                f.write(f"{failure['subject']}: {failure['reason']}\n")
        print(f"\n{len(failed_subjects)} failed subjects logged to failed_subjects.txt")

if __name__ == '__main__':
    main()
        


