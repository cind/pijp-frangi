import os
import subprocess
import numpy as np
import nibabel as nib
import shutil
import pandas as pd
import argparse
# from nipype.interfaces.spm import NewSegment
# from nipype.interfaces.spm.base import Info
import gzip
from pathlib import Path
#from nipype.interfaces.matlab import MatlabCommand
# tells nipype which MATLAB executable to use and where SPM12 is
# MatlabCommand.set_default_matlab_cmd('matlab -nodisplay -nosplash')
# MatlabCommand.set_default_paths('/opt/mathworks/MatlabToolkits/spm12_r7219')

# test matlab and spm
#print("MATLAB cmd:", MatlabCommand().cmd)
# print("SPM version:", NewSegment().version)
# print("SPM path:", Info.name())

# # workaround for the docstring version-check bug
# import nipype.interfaces.spm as spm
# spm.SPMCommand.set_mlab_paths(
#     matlab_cmd='matlab -nodisplay -nosplash',
#     use_mcr=False
# )

os.environ['SPMPATH'] = '/opt/mathworks/MatlabToolkits/spm12_r7219'
#os.environ['MATLAB_VERSION'] = 'R2019a'
##########################################
###     some useful functions       ####
##########################################

def run_command(cmd_list):
    """Run a command and raise error if it fails"""
    subprocess.run(cmd_list, check=True)


# def spm12_brain_extract(t1_path, spm12_dir, output_mask, output_brain):    
#     # Configure SPM12 tissue probability maps
#     tpm = spm12_dir + '/tpm/TPM.nii'
#     tissue_list = []
#     for i, (ngaus, native) in enumerate([
#         (1, (1, 0)),   # GM
#         (1, (1, 0)),   # WM
#         (2, (1, 0)),   # CSF
#         (3, (0, 0)),   # skull   — excluded
#         (4, (0, 0)),   # soft tissue — excluded
#         (2, (0, 0)),   # air/background — excluded
#     ], start=1):
#         tissue_list.append(((tpm, i), ngaus, native, (0, 0)))

#     # Run segmentation
#     seg = NewSegment()
#     seg.inputs.channel_files = t1_path
#     seg.inputs.channel_info = (0, np.inf, (False, False))  # skip bias correction
#     seg.inputs.tissues = tissue_list
#     seg.inputs.write_deformation_fields = [False, False]
#     result = seg.run()

#     # Build brain mask from GM + WM + CSF probability maps
#     prob_maps = result.outputs.native_class_images[:3]  # classes 1-3 only
#     ref = nib.load(t1_path)
#     brain_prob = sum(nib.load(p[0]).get_fdata() for p in prob_maps)
#     mask = (brain_prob > 0.5).astype(np.int16)
#     stripped = ref.get_fdata() * mask

#     # Save outputs
#     # stem = Path(t1_path).stem.replace('.nii', '')
#     mask_path = output_mask
#     brain_path = output_brain
#     nib.save(nib.Nifti1Image(mask, ref.affine, ref.header), mask_path)
#     nib.save(nib.Nifti1Image(stripped, ref.affine, ref.header), brain_path)

#     return brain_path


def spm12_brain_extract(t1_path, spm12_dir, output_mask, output_brain, export_matlab_version):
    
    matlab_script = f"""
try
    addpath('{spm12_dir}');
    spm('defaults', 'FMRI');
    spm_jobman('initcfg');

    matlabbatch{{1}}.spm.spatial.preproc.channel.vols = {{'{t1_path},1'}};
    matlabbatch{{1}}.spm.spatial.preproc.channel.biasreg = 0;
    matlabbatch{{1}}.spm.spatial.preproc.channel.biasfwhm = Inf;
    matlabbatch{{1}}.spm.spatial.preproc.channel.write = [0 0];
    matlabbatch{{1}}.spm.spatial.preproc.tissue(1).tpm = {{'{spm12_dir}/tpm/TPM.nii,1'}};
    matlabbatch{{1}}.spm.spatial.preproc.tissue(1).ngaus = 1;
    matlabbatch{{1}}.spm.spatial.preproc.tissue(1).native = [1 0];
    matlabbatch{{1}}.spm.spatial.preproc.tissue(1).warped = [0 0];
    matlabbatch{{1}}.spm.spatial.preproc.tissue(2).tpm = {{'{spm12_dir}/tpm/TPM.nii,2'}};
    matlabbatch{{1}}.spm.spatial.preproc.tissue(2).ngaus = 1;
    matlabbatch{{1}}.spm.spatial.preproc.tissue(2).native = [1 0];
    matlabbatch{{1}}.spm.spatial.preproc.tissue(2).warped = [0 0];
    matlabbatch{{1}}.spm.spatial.preproc.tissue(3).tpm = {{'{spm12_dir}/tpm/TPM.nii,3'}};
    matlabbatch{{1}}.spm.spatial.preproc.tissue(3).ngaus = 2;
    matlabbatch{{1}}.spm.spatial.preproc.tissue(3).native = [1 0];
    matlabbatch{{1}}.spm.spatial.preproc.tissue(3).warped = [0 0];
    matlabbatch{{1}}.spm.spatial.preproc.tissue(4).tpm = {{'{spm12_dir}/tpm/TPM.nii,4'}};
    matlabbatch{{1}}.spm.spatial.preproc.tissue(4).ngaus = 3;
    matlabbatch{{1}}.spm.spatial.preproc.tissue(4).native = [0 0];
    matlabbatch{{1}}.spm.spatial.preproc.tissue(4).warped = [0 0];
    matlabbatch{{1}}.spm.spatial.preproc.tissue(5).tpm = {{'{spm12_dir}/tpm/TPM.nii,5'}};
    matlabbatch{{1}}.spm.spatial.preproc.tissue(5).ngaus = 4;
    matlabbatch{{1}}.spm.spatial.preproc.tissue(5).native = [0 0];
    matlabbatch{{1}}.spm.spatial.preproc.tissue(5).warped = [0 0];
    matlabbatch{{1}}.spm.spatial.preproc.tissue(6).tpm = {{'{spm12_dir}/tpm/TPM.nii,6'}};
    matlabbatch{{1}}.spm.spatial.preproc.tissue(6).ngaus = 2;
    matlabbatch{{1}}.spm.spatial.preproc.tissue(6).native = [0 0];
    matlabbatch{{1}}.spm.spatial.preproc.tissue(6).warped = [0 0];
    matlabbatch{{1}}.spm.spatial.preproc.warp.mrf = 1;
    matlabbatch{{1}}.spm.spatial.preproc.warp.cleanup = 1;
    matlabbatch{{1}}.spm.spatial.preproc.warp.reg = [0 0.001 0.5 0.05 0.2];
    matlabbatch{{1}}.spm.spatial.preproc.warp.affreg = 'mni';
    matlabbatch{{1}}.spm.spatial.preproc.warp.fwhm = 0;
    matlabbatch{{1}}.spm.spatial.preproc.warp.samp = 3;
    matlabbatch{{1}}.spm.spatial.preproc.warp.write = [0 0];

    spm_jobman('run', matlabbatch);
catch ME
    report = ME.getReport;
    fprintf(2, report);
    exit(-1);
end
exit;"""

    # Write to a temp .m file, mirroring your working script pattern
    script_path = os.path.join(os.path.dirname(t1_path), 'spm_segment.m')
    with open(script_path, 'w') as f:
        f.write(matlab_script)

    cmd = f'export MATLAB_VERSION={export_matlab_version} && matlab -singleCompThread -nodesktop -noFigureWindows -nojvm -nosplash -r spm_segment'
    proc = subprocess.Popen(cmd, shell=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            cwd=os.path.dirname(t1_path))
    output, error = proc.communicate()

    if proc.returncode != 0:
        error = error.decode('ascii', errors='ignore')
        raise RuntimeError(f'MATLAB/SPM failure:\n{error}')

    # Build brain mask from GM + WM + CSF probability maps
    t1_stem = os.path.basename(t1_path).replace('.nii', '')
    t1_dir = os.path.dirname(t1_path)
    ref = nib.load(t1_path)
    brain_prob = sum(
        nib.load(os.path.join(t1_dir, f'c{i}{t1_stem}.nii')).get_fdata()
        for i in range(1, 4)
    )
    mask = (brain_prob > 0.5).astype(np.int16)
    stripped = ref.get_fdata() * mask

    nib.save(nib.Nifti1Image(mask, ref.affine, ref.header), output_mask)
    nib.save(nib.Nifti1Image(stripped, ref.affine, ref.header), output_brain)

    return output_brain

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
        subjname = Path(t1).stem.replace('.T1.nii', '')   # redefine subject based on full image name
        shutil.copy(t1,os.path.join(output_dir,subjname + '-T1raw.nii.gz'))
        t1 = os.path.join(output_dir, subjname + '-T1raw.nii.gz')
        print('t1 was copied over')

        flair = [os.path.join(subj_dir,flair) for flair in os.listdir(subj_dir) if flair.endswith('.FLAIR.nii.gz')][0]
        shutil.copy(flair,os.path.join(output_dir,subjname + '-FLAIRraw.nii.gz'))
        flair = os.path.join(output_dir, subjname + '-FLAIRraw.nii.gz')
        print('flair was copied over')
        
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
        print('t1 bias field correction finished')

        ## identify raw flair and processing with N4 bias field correction
        # flair_bc = os.path.join(subj_dir,subject+'-FLAIRbc.nii.gz')
        flair_bc = os.path.join(output_dir,subjname+'-FLAIRbc.nii.gz')
        run_command(['N4BiasFieldCorrection', '-i', flair, '-o', flair_bc])
        print('flair bias field correction finished')

        # #### register flair to t1 and to template:
        flair_bcreg = os.path.join(output_dir,subjname+'-FLAIRbcreg.nii.gz')
        run_command(['flirt', '-in', flair_bc, '-ref',t1_bc,'-out',flair_bcreg, '-dof', '6'])
        print('flair registered to t1')

        # ### brain extraction using SPM
        # t1_bc_brainextract = os.path.join(output_dir,subjname+'-T1bcbrainmask.nii.gz')
        # brain_mask = os.path.join(output_dir,subjname+'-brainmask.nii.gz')
        # print("SPM setup complete, starting segmentation...")
        # unzipped_t1 = gunzip(t1_bc)
        # spm12_brain_extract(unzipped_t1,spm12_path,brain_mask,t1_bc_brainextract,'R2019a')
        # print("Segmentation complete")


        ## intensity normalization with fuzzy-C means: https://github.com/jcreinhold/intensity-normalization?tab=readme-ov-file
        t1_bc_brainextract_norm = os.path.join(output_dir,subjname+'-T1bcbrainmask_norm.nii.gz')

        ## need this becuase python version is too low
        #intensity_norm_python = '/home/vhasfctangs1/pijp-frangi/pijp_frangi/normenv/bin/python'
        intensity_norm_exe = '/home/vhasfctangs1/pijp-frangi/normvenv/bin/zscore-normalize'
        #run_command([intensity_norm_exe, t1_bc_brainextract, '-o', t1_bc_brainextract_norm])
        run_command([intensity_norm_exe, t1_bc, '-o', t1_bc_brainextract_norm])
        #run_command(['intensity-normalize', 'zscore', t1_bc_brainextract, '-o', t1_bc_brainextract_norm])
        #run_command([intensity_norm_python, '-m', 'intensity_normalization.cli.zscore', t1_bc_brainextract, '-o', t1_bc_brainextract_norm])
        print("finished intensity normalization")

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
        


