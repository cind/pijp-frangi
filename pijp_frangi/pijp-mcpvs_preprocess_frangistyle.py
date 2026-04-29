import os
import logging
import argparse
import datetime
import subprocess
import nibabel as nib
import gzip
from pathlib import Path
import shutil

from pijp import util
from pijp.repositories import ProcessingLog
from pijp.core import Step, get_project_dir
from pijp.engine import run_module, run_file
from pijp.exceptions import ProcessingError

from frangi import BaseStep, Commands

LOGGER = logging.getLogger(__name__)
PROCESS_TITLE = 'mcpvs_preprocessv6'

def get_process_dir(project):
    return os.path.join(get_project_dir(project), PROCESS_TITLE)

def get_case_dir(project, code):
    cdir = os.path.join(get_process_dir(project), code)
    if not os.path.isdir(cdir):
        os.makedirs(cdir)
    return cdir


class PreprocessSubject(BaseStep):
    process_name = PROCESS_TITLE
    step_name = 'preprocess'
    step_cli = 'preprocess'
    cpu = 2
    mem = '16G'

    def __init__(self, project, code, args):
        super().__init__(project, code, args)
        self.original_code = code
        self.commands = Commands(project, code, args)

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


    def spm12_brain_extract(self, t1_path, spm12_dir, output_mask, output_brain, export_matlab_version):

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

    def register(self,input,ref,output):
        cmd = f'flirt -in {input} -ref {ref} -out {output}'
        self.commands.fsl(cmd)

    def N4Bias(self,input,output):
        cmd = f'N4BiasFieldCorrection -i {input} -o {output}'
        self.commands.ants(cmd)
    
    def intensitynorm(self,input,output):
        normcommand = '/home/vhasfctangs1/pijp-frangi/normvenv/bin/fcm-normalize'
        cmd = f'{normcommand} {input} -o {output}'
        self._run_cmd(cmd,script_name='norm_func')


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

        #os.makedirs(output_dir,exist_ok=True)    # in case it doesn't exist

        # find the t1s and stuff
        t1 = [os.path.join(self.subj_dir,t1) for t1 in os.listdir(self.subj_dir) if t1.endswith('.T1.nii.gz')][0]
        subjname = Path(t1).stem.replace('.T1.nii', '')   # redefine subject based on full image name
        shutil.copy(t1,os.path.join(self.output_folder,subjname + '-T1raw.nii.gz'))
        self.t1raw = os.path.join(self.output_folder, subjname + '-T1raw.nii.gz')
        print('t1 was copied over')

        flair = [os.path.join(self.subj_dir,flair) for flair in os.listdir(self.subj_dir) if flair.endswith('.FLAIR.nii.gz')][0]
        shutil.copy(flair,os.path.join(self.output_folder,subjname + '-FLAIRraw.nii.gz'))
        self.flairraw = os.path.join(self.output_folder, subjname + '-FLAIRraw.nii.gz')
        print('flair was copied over')
        
        # Check T1 exists
        if not os.path.exists(t1):
            raise FileNotFoundError(f"T1 not found: {t1}")
        
        # Check flair exists
        if not os.path.exists(flair):
            raise FileNotFoundError(f"flair not found: {flair}")

        # Get the full path to the preprocessing script
        # Assumes it's in the same directory as this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        spm12_path = '/opt/mathworks/MatlabToolkits/spm12_r7219'

        t1_bc = os.path.join(self.output_folder,subjname+'-T1bc.nii.gz')
        self.N4Bias(self.t1raw, t1_bc)

        flair_bc = os.path.join(self.output_folder,subjname+'-FLAIRbc.nii.gz')
        self.N4Bias(self.flairraw, flair_bc)

        flair_bcreg = os.path.join(self.output_folder,subjname+'-FLAIRbcreg.nii.gz')
        self.register(flair_bc,t1_bc,flair_bcreg)

        t1_bc_brainextract = os.path.join(self.output_folder,subjname+'-T1bcbrainmask.nii.gz')
        brain_mask = os.path.join(self.output_folder,subjname+'-brainmask.nii.gz')
        matlab_script = self.spm12_brain_extract(t1_bc, spm12_path, brain_mask, t1_bc_brainextract)
        self.matlab(matlab_script)

        t1_bc_brainextract_norm = os.path.join(self.output_folder,subjname+'-T1bcbrainmask_norm.nii.gz')
        self.intensitynorm(t1_bc_brainextract,t1_bc_brainextract_norm)


def run():
    import sys
    current_module = sys.modules[__name__]
    run_module(current_module)


if __name__ == "__main__":
    run_file(os.path.abspath(__file__))
