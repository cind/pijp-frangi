import os
import subprocess
import numpy as np
import nibabel as nib
import shutil
import pandas as pd
import argparse

##########################################
###     some useful functions       ####
##########################################

def run_command(cmd_list):
    """Run a command and raise error if it fails"""
    subprocess.run(cmd_list, check=True)

def get_crop_coords(input_file, target_size=200):
    """Calculate crop coordinates from one reference image"""
    img = nib.load(input_file)
    data = img.get_fdata()
    
    x, y, z = data.shape[:3]
    center_x, center_y, center_z = x // 2, y // 2, z // 2
    
    half_size = target_size // 2
    
    x_start = max(0, center_x - half_size)
    x_end = min(x, center_x + half_size)
    y_start = max(0, center_y - half_size)
    y_end = min(y, center_y + half_size)
    z_start = max(0, center_z - half_size)
    z_end = min(z, center_z + half_size)
    
    return (x_start, x_end, y_start, y_end, z_start, z_end)

def apply_crop(input_file, output_file, crop_coords):
    """Apply pre-calculated crop coordinates"""
    x_start, x_end, y_start, y_end, z_start, z_end = crop_coords
    
    img = nib.load(input_file)
    data = img.get_fdata()
    
    cropped_data = data[x_start:x_end, y_start:y_end, z_start:z_end]
    
    affine = img.affine.copy()
    affine[:3, 3] += affine[:3, :3] @ np.array([x_start, y_start, z_start])
    
    cropped_img = nib.Nifti1Image(cropped_data, affine, img.header)
    nib.save(cropped_img, output_file)

# wont need here but keeping for reference
def uncrop_with_coords(cropped_file, original_file, output_file, crop_coords):
    """Uncrop using the exact coordinates that were used for cropping"""
    x_start, x_end, y_start, y_end, z_start, z_end = crop_coords
    
    cropped_img = nib.load(cropped_file)
    original_img = nib.load(original_file)
    
    cropped_data = cropped_img.get_fdata()
    orig_shape = original_img.shape[:3]
    
    # Create output array filled with zeros
    uncropped_data = np.zeros(orig_shape)
    
    # Place cropped data at EXACT original location
    uncropped_data[x_start:x_end, y_start:y_end, z_start:z_end] = cropped_data
    
    # Use original affine and header
    uncropped_img = nib.Nifti1Image(uncropped_data, original_img.affine, original_img.header)
    nib.save(uncropped_img, output_file)
# Usage
#uncrop_to_original('T1_cropped.nii.gz', 'T1.nii.gz', 'T1_uncropped.nii.gz')#




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
    
    # Now you can use the arguments
    # parent_dir = args.img_folder
    # talairach_paths = args.talairach_folder
    # flair_raw_paths = args.raw_folder
    
    output_dir = args.output_folder
    subj_dir = args.subj_dir
    subject = args.subject
    
    print(f"Processing images in: {subj_dir}")
    #print(f"Output directory: {output_dir}")
    failed_subjects = []  # Add this before your loop
    talairach_paths = '/m/InProcess/External/ADNI3_FSdn/Freesurfer/subjects'
    flair_raw_paths = '/m/InProcess/External/ADNI3/ADNI3_frangi/Raw'
    
    try:
        #subject = subj_dir.split('/')[-1]
        os.makedir(output_dir,exist_ok=True)    # in case it doesn't exist
        # files I need: t1, talairach, raw flair, wmmask
        t1 = os.path.join(subj_dir, subject + '-T1.nii.gz')
        shutil.copy(t1,output_dir)
        t1 = os.path.join(output_dir, subject + '-T1.nii.gz')
        wmmask = os.path.join(subj_dir, subject + '-wmmask.nii.gz')
        shutil.copy(wmmask,output_dir)
        wmmask = os.path.join(output_dir, subject + '-wmmask.nii.gz')
        
        # Check T1 exists
        if not os.path.exists(t1):
            raise FileNotFoundError(f"T1 not found: {t1}")
        
        # Check wmmask exists
        if not os.path.exists(wmmask):
            raise FileNotFoundError(f"WM mask not found: {wmmask}")
        
        # Get talairach
        # can i be more specific?
        talairach_subjfolder = [t for t in os.listdir(talairach_paths) if subject == t]
        if not talairach_subjfolder:
            raise FileNotFoundError(f"No talairach folder found for {subject[:-9]}")
        
        # talairach_xfm = os.path.join(talairach_paths, talairach_subjfolder[0], 'mri', 'transforms', 'talairach.xfm')
        # if not os.path.exists(talairach_xfm):
        #     raise FileNotFoundError(f"Talairach.xfm not found: {talairach_xfm}")
        # talairach = os.path.join(subj_dir, subject + '-talairach.xfm')
        # shutil.copy(talairach_xfm, talairach)

        talairach_lta = os.path.join(talairach_paths, talairach_subjfolder[0], 'mri', 'transforms', 'talairach.lta')
        if not os.path.exists(talairach_lta):
            raise FileNotFoundError(f"Talairach.lta not found: {talairach_lta}")
        talairach = os.path.join(subj_dir, subject + '-talairach.lta')
        shutil.copy(talairach_lta, talairach)
        shutil.copy(talairach,output_dir)
        talairach = os.path.join(output_dir, subject + '-talairach.lta')

        
        # Get FLAIR
        flair_raw_subjfolder = [f for f in os.listdir(flair_raw_paths) if subject[0:19] in f and 'long' not in f]
        if not flair_raw_subjfolder:
            flair_raw_subjfolder = [f for f in os.listdir(flair_raw_paths) if subject[0:17] in f and 'long' not in f]
            if not flair_raw_subjfolder:
                raise FileNotFoundError(f"No FLAIR folder found for {subject[:-9]}")
        
        flair_candidates = [os.path.join(flair_raw_paths, flair_raw_subjfolder[0], f) 
                            for f in os.listdir(os.path.join(flair_raw_paths, flair_raw_subjfolder[0])) 
                            if '.FLAIR.nii.gz' in f]
        if not flair_candidates:
            raise FileNotFoundError(f"No FLAIR.nii.gz found in {os.path.join(flair_raw_paths, flair_raw_subjfolder[0])}")
        
        flair_source = flair_candidates[0]
        if not os.path.exists(flair_source):
            raise FileNotFoundError(f"FLAIR not found: {flair_source}")
        
        flair = os.path.join(subj_dir, subject + '-FLAIRraw.nii.gz')
        shutil.copy(flair_source, flair)
        shutil.copy(flair,output_dir)
        flair = os.path.join(output_dir, subject + '-FLAIRraw.nii.gz')

    # print('t1:',t1)
    # print('talairach:',talairach)
    # print('flair:',flair)
    

        ##########################################
        #####           processing            ####
        ##########################################
        
        ## mni coords
        mni305cor = '/opt/freesurfer/7.4.1/average/mni305.cor.mgz'

        # convert t1 to template space
        # t1_template = os.path.join(subj_dir,subject+'-T1_template.nii.gz')
        t1_template = os.path.join(output_dir,subject+'-T1_template.nii.gz')
        run_command(['mri_convert', t1, '--apply_transform', talairach, '-rt', 'cubic', '-rl', mni305cor, '-odt','float', t1_template])

        ## identify raw flair and processing with N4 bias field correction
        # flair_bc = os.path.join(subj_dir,subject+'-FLAIRbc.nii.gz')
        flair_bc = os.path.join(output_dir,subject+'-FLAIRbc.nii.gz')
        run_command(['N4BiasFieldCorrection', '-i', flair, '-o', flair_bc])

        #### register flair to t1 and to template:
        ## register flair to t1, but only keep the matrix (6 DOF, using mri_coreg to stay in FS environment)
        #flair2t1_mat =  os.path.join(subj_dir,subject+'-flair2t1.lta')
        flair2t1_mat =  os.path.join(output_dir,subject+'-flair2t1.lta')
        run_command(['mri_coreg', '--mov',flair_bc, '--ref', t1, '--lta',flair2t1_mat,'--dof','6'])

        ## concat the flair to t1 matrix and the talairach matrix
        #flair2template_lta = os.path.join(subj_dir,subject+'-flair2template.lta')
        flair2template_lta = os.path.join(output_dir,subject+'-flair2template.lta')
        run_command(['mri_concatenate_lta', flair2t1_mat, talairach, flair2template_lta])

        ## apply the concatenated matrix (using mri_convert now)
        # flair_bcreg_template = os.path.join(subj_dir,subject+'-FLAIRbcreg_template.nii.gz')
        flair_bcreg_template = os.path.join(output_dir,subject+'-FLAIRbcreg_template.nii.gz')
        run_command(['mri_convert',flair_bc, '--apply_transform', flair2template_lta, '-rt','cubic','-rl',mni305cor, '-odt','float',flair_bcreg_template])


        #### normalize flair
        ## normalize flair by first applying talairach to WM mask (using mri_convert again)
        #wmmask_template = os.path.join(subj_dir,subject+'-wmmask_template.nii.gz')
        wmmask_template = os.path.join(output_dir,subject+'-wmmask_template.nii.gz')
        run_command(['mri_convert', wmmask, '--apply_transform', talairach, '-rt','nearest','-rl',mni305cor,'-odt','float',wmmask_template])

        ## then do FS linear shift, with WM=1000 
        flair_template_data = nib.load(flair_bcreg_template).get_fdata()
        wmmask_template_data = nib.load(wmmask_template).get_fdata()
        # flair_mean = np.mean(flair_template_data*wmmask_template_data)
        wm_voxels = flair_template_data[wmmask_template_data > 0]
        flair_mean = np.mean(wm_voxels)
        scale_factor = 1000 / flair_mean
        #flair_bcreg_template_norm = os.path.join(subj_dir,subject+'-FLAIRbcreg_template_std.nii.gz')        ## std = standardized, not to be confused with normalized (if i decide to z-score norm later)
        flair_bcreg_template_norm = os.path.join(output_dir,subject+'-FLAIRbcreg_template_std.nii.gz')   
        run_command(['fslmaths', flair_bcreg_template, '-mul', str(scale_factor), flair_bcreg_template_norm])

        ## crop to 200 around center
        ## only doing this for training; bring back to original size so the ROI masks fit
        #t1_template_cropped = os.path.join(subj_dir,subject+'-T1_template_crop.nii.gz')
        #flair_bcreg_template_norm_cropped = os.path.join(subj_dir,subject+'-FLAIRbcreg_template_std_crop.nii.gz')

        t1_template_cropped = os.path.join(output_dir,subject+'-T1_template_crop.nii.gz')
        flair_bcreg_template_norm_cropped = os.path.join(output_dir,subject+'-FLAIRbcreg_template_std_crop.nii.gz')

        crop_coords = get_crop_coords(t1_template)
        apply_crop(t1_template, t1_template_cropped, crop_coords)
        apply_crop(flair_bcreg_template_norm, flair_bcreg_template_norm_cropped, crop_coords)

        #### z-score not required, built into nnunet pipeline

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
        


