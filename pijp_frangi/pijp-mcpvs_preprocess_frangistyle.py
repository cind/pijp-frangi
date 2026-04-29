import os
import logging
import argparse
import datetime
import subprocess
import nibabel as nib
import frangi

from pijp import util
from pijp.repositories import ProcessingLog
from pijp.core import Step, get_project_dir
from pijp.engine import run_module, run_file
from pijp.exceptions import ProcessingError

LOGGER = logging.getLogger(__name__)
PROCESS_TITLE = 'mcpvs_preprocessv6'

def get_process_dir(project):
    return os.path.join(get_project_dir(project), PROCESS_TITLE)

def get_case_dir(project, code):
    cdir = os.path.join(get_process_dir(project), code)
    if not os.path.isdir(cdir):
        os.makedirs(cdir)
    return cdir


class PreprocessSubject(Step):
    process_name = PROCESS_TITLE
    step_name = 'preprocess'
    step_cli = 'preprocess'
    cpu = 2
    mem = '16G'
    
    def __init__(self, project, code, args):
        self.original_code = code 

        # Parse research group and subject from the full path
        if '/' in code:
            parts = code.rstrip('/').split('/')
            #research_group = parts[-2]
            subject = parts[-1]
        else:
            # # Fallback if only subject name is passed
            # safe_code = code
            # research_group = "UNKNOWN"
            # subject = code
            subject = code


        super().__init__(project, code, args)
        self.datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
        self.project = project
        #self.research_group = research_group
        self.subject = subject
        self.code = self.subject

        # Use original_code if available, otherwise reconstruct
        if '/' in self.original_code:
            self.subj_dir = self.original_code
        else:
            # Need to reconstruct from args or stored info
            parent_dir = '/m/Researchers/SerenaT/deeppvs/for_nnunet/groundtruth_rawimgs'
            self.subj_dir = os.path.join(parent_dir, self.subject)

        LOGGER.info(f"Received code: {code}")
        LOGGER.info(f"Original code: {self.original_code}")
       # LOGGER.info(f"Research group: {self.research_group}, Subject: {self.code}")
        LOGGER.info(f"Subject directory: {self.subj_dir}")

        self.working_dir = get_case_dir(self.project, self.code)
        self.output_folder = os.path.join(
            '/m/Researchers/SerenaT/deeppvs/for_nnunet/gt_mcpvs_preprocessed3',
            self.code
        )
        
    @classmethod
    def get_queue(cls, project_name):
        """
        Build the queue of all subjects to process.
        Returns a list of dicts with ProjectName and Code (subject path).
        """
        parent_dir = '/m/Researchers/SerenaT/deeppvs/for_nnunet/groundtruth_rawimgs'
        #dx_names = ['EMCI', 'AD', 'MCI', 'CN', 'LMCI', 'SMC']
        
        # Get already attempted subjects
        attempted_rows = ProcessingLog().get_step_attempted(project_name, PROCESS_TITLE, 'preprocess')
        attempted = [row[1] for row in attempted_rows]
        
        todo = []
        # for research_group in dx_names:
        #     dx_dir = os.path.join(parent_dir, research_group)
        #     if not os.path.isdir(dx_dir):
        #         LOGGER.warning(f"Directory not found: {dx_dir}")
        #         continue
            
        # Get all subject folders
        subjects = [s for s in os.listdir(parent_dir) 
                    if not s.startswith('.') and os.path.isdir(os.path.join(parent_dir, s))]
        
        for subject in subjects:
            subj_path = os.path.join(parent_dir, subject)
            if subj_path not in attempted:
                todo.append({
                    'ProjectName': project_name,
                    'Code': subj_path  # Full path to subject folder
                })
        
        LOGGER.info(f"Found {len(todo)} subjects to process")
        return todo
    
    def run(self):
        """
        Main processing entry point.
        """
        os.environ['FSVERSION'] = '7.4.1'
        os.environ['ANTSVERSION'] = 'ants-2.5.0'
        os.environ['FSLVERSION'] = '6.0.0'
        os.environ['MATLAB_VERSION'] = 'R2019a'
        LOGGER.info(f"Processing subject: {self.subject}")
        LOGGER.info(f"Subject directory: {self.subj_dir}")
        LOGGER.info(f"Output folder: {self.output_folder}")
        
        # Create output directory
        os.makedirs(self.output_folder, exist_ok=True)
        
        # Get the full path to the preprocessing script
        # Assumes it's in the same directory as this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        preprocess_script = os.path.join(script_dir, 'grid_mcpvs_preprocessing.py')
        
        # Run your preprocessing Python script
        import sys
        python_exe = sys.executable
        cmd = [
            python_exe,
            preprocess_script,
            '--subj_dir', self.subj_dir,
            '--subject', self.subject,
            '--output_folder', self.output_folder
        ]
        LOGGER.info(f"Using Python: {python_exe}")
        LOGGER.info(f"Running command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding='utf-8',
                check=True
            )
            
            if result.stdout:
                LOGGER.info(result.stdout)
            if result.stderr:
                LOGGER.warning(result.stderr)
                
            LOGGER.info(f"Successfully processed {self.subject}")
            
        except subprocess.CalledProcessError as e:
            LOGGER.error(f"Processing failed for {self.subject}")
            LOGGER.error(f"stdout: {e.stdout}")
            LOGGER.error(f"stderr: {e.stderr}")
            self.outcome = 'Error'
            self.comments = f"Preprocessing failed: {e.stderr}"
            raise ProcessingError(f"Preprocessing failed for {self.subject}")


class PreprocessingMethods(frangi):
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

        # Write to a temp .m file, mirroring frangi script patterns
        script_path = os.path.join(os.path.dirname(t1_path), 'spm_segment.m')
        with open(script_path, 'w') as f:
            f.write(matlab_script)

        # export includes extra stuff to prevent screens from popping up / memory issues
        #cmd = f'export MATLAB_VERSION={export_matlab_version} && matlab -singleCompThread -nodesktop -noFigureWindows -nojvm -nosplash -r spm_segment'
        #cmd = f'export MATLAB_VERSION={export_matlab_version} && export MATLAB_USE_USERWORK=0 && export MW_DDUX_DISABLE=1 && matlab -singleCompThread -nodesktop -noFigureWindows -nojvm -nosplash -r spm_segment'
        cmd = (
        f'export MATLAB_VERSION={export_matlab_version} && '
        f'export MW_DDUX_DISABLE=1 && '
        f'export MW_CONNECTOR_ENABLE=false && '
        f'export MATLAB_ENABLE_NETWORK=0 && '
        f'export HTTP_PROXY="" && '
        f'export HTTPS_PROXY="" && '
        f'matlab -singleCompThread -nodesktop -noFigureWindows -nojvm -nosplash -nodisplay -r spm_segment'
            )
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

            ### brain extraction using SPM
            t1_bc_brainextract = os.path.join(output_dir,subjname+'-T1bcbrainmask.nii.gz')
            brain_mask = os.path.join(output_dir,subjname+'-brainmask.nii.gz')
            print("SPM setup complete, starting segmentation...")
            unzipped_t1 = gunzip(t1_bc)
            spm12_brain_extract(unzipped_t1,spm12_path,brain_mask,t1_bc_brainextract,'R2019a')
            print("Segmentation complete")


            ## intensity normalization with fuzzy-C means: https://github.com/jcreinhold/intensity-normalization?tab=readme-ov-file
            t1_bc_brainextract_norm = os.path.join(output_dir,subjname+'-T1bcbrainmask_norm.nii.gz')

            ## need this becuase python version is too low
            #intensity_norm_python = '/home/vhasfctangs1/pijp-frangi/pijp_frangi/normenv/bin/python'
            #run_command([intensity_norm_exe, t1_bc, '-o', t1_bc_brainextract_norm])
            #run_command(['intensity-normalize', 'zscore', t1_bc_brainextract, '-o', t1_bc_brainextract_norm])
            #run_command([intensity_norm_python, '-m', 'intensity_normalization.cli.zscore', t1_bc_brainextract, '-o', t1_bc_brainextract_norm])
            
            intensity_norm_exe = '/home/vhasfctangs1/pijp-frangi/normvenv/bin/fcm-normalize'
            run_command([intensity_norm_exe, t1_bc_brainextract, '-o', t1_bc_brainextract_norm])
            print("finished intensity normalization")

            #### everything works except the matlab part

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



def run():
    import sys
    current_module = sys.modules[__name__]
    run_module(current_module)


if __name__ == "__main__":
    run_file(os.path.abspath(__file__))