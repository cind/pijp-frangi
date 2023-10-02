import os
import subprocess
import logging
import argparse
import textwrap
import string

import numpy as np
import shutil
import pandas as pd
import sys
import gzip
import nibabel as nib
from matplotlib import pyplot as plt
from scipy import stats

# Before loading NUMPY: HACK to avoid seg fault on some hosts.
#os.environ["OMP_NUM_THREADS"] = "1"

import repo
from pijp import util, coding
from pijp.repositories import ProcessingLog
from pijp.core import Step, get_project_dir
from pijp.engine import run_module, run_file
from pijp.exceptions import ProcessingError, NoLogProcessingError


LOGGER = logging.getLogger(__name__)
PROCESS_TITLE = 'pijp-frangi'

def get_process_dir(project):
    return os.path.join(get_project_dir(project), PROCESS_TITLE)

def get_case_dir(project, researchgroup, code):
    cdir = os.path.join(get_process_dir(project), researchgroup, code)
    if not os.path.exists(cdir):
        os.makedirs(cdir)

    return cdir


class BaseStep(Step):
    def __init__(self, project, code, args):
        super().__init__(project, code, args)
        # Put all of your name space variables here, e.g. image file names.
        self.data = os.path.join(get_project_dir('ADNI3_FSdn'),'Freesurfer','subjects')   # this will be changed when i run this on Raw: ADNI3_FSdn/Raw
        self.project = project
        self.code = code

        # other folders:
        researchgroup = repo.Repository(self.project).get_researchgroup(self.code)
        self.working_dir = get_case_dir(self.project, researchgroup, self.code)

        # only for freesurfer
        self.mrifolder = os.path.join(self.data,self.code,'mri')
        self.statsfolder = os.path.join(self.data,self.code,'stats')

        # all nii converted
        self.t1 = os.path.join(self.working_dir, self.code + "-T1.nii.gz")
        self.wmdgmask = os.path.join(self.working_dir, self.code + "-wmdgmask.nii.gz")
        self.wmparcmask = os.path.join(self.working_dir, self.code + "-wmmask.nii.gz")
        self.frangimask = os.path.join(self.working_dir, self.code + "-frangi-thresholded.nii.gz")
        self.asegstats = os.path.join(self.working_dir, self.code + "-asegstats.csv")
	
        # If you need to get time point, site id, subject number, or other meta data
        # that is stored in the series code, use this object.
        self.Code = coding.Code(self.code)
        # You may want a scan code to lookup other image types from the same visit.
        self.scan_code = self.Code.scan_code



    def _run_cmd(self, cmd, script_name=None):
        if script_name is None:
            script_name = f"{self.datetime}-{self.__class__.__name__}.sh"
        else:
            script_name = f"{self.datetime}-{script_name}.sh"

        LOGGER.debug(cmd)
        script_path = self._prep_cmd_script(cmd, script_name)
        output = self._run_script(script_path)
        return output

    def _prep_cmd_script(self, cmd, script_name):
        cmd = textwrap.dedent(cmd)
        script_template = string.Template("$cmd")
        script = script_template.safe_substitute({'cmd': cmd})

        script_path = os.path.join(self.working_dir, 'scripts', script_name)
        if not os.path.exists(os.path.dirname(script_path)):
            os.makedirs(os.path.dirname(script_path))

        with open(script_path, 'w') as f:
            f.write(script)

        os.chmod(script_path, 0o777)

        return script_path

    def _run_script(self, script_path):
        p = subprocess.Popen(script_path,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             encoding="utf-8",
                             errors='ignore',
                             shell=True)

        LOGGER.debug(f"running script:{script_path}")

        output, error = p.communicate()
        if len(output) > 0:
            LOGGER.debug(output)

        if len(error) > 0 and p.returncode == 0:
            LOGGER.warning(error)

        if p.returncode != 0:
            LOGGER.error(output)
            LOGGER.error(error)

        if p.returncode != 0:
            lines = []

            if output:
                lines = output.splitlines()
            if error:
                lines = lines + error.splitlines()

            if len(lines) > 20:
                lines = lines[-20]

            self.comments = (self.comments or "") + "\n".join(lines)
            self.outcome = 'Error'
            raise ProcessingError(f"Script Failed! {script_path}")

        return output


class Commands(BaseStep):
    # example:
    # commands.fs(func) where func = f'some command -i {input} -o {output}, then input and output are specified at the actual functions --> at the moment, I have it so this already includes running the command 

    exportfs = '6.0.0'
    exportants = 'ants-2017-12-07'
    exportfsl = '6.0.0'
    exportmatlab = 'R2019a'
    #qitpath = 'qit'

    def qit(self,func):
        cmd = f'qit {func}' 
        _run_cmd(self,cmd,script_name=func)

        
    def fs(self,func):
        cmd = f'export FSVERSION={self.exportfs} && {func}'
        _run_cmd(self,cmd,script_name=func)
    
    def ants(self,func):
        cmd = f'export ANTSVERSION= {self.exportants} && {func}'
        _run_cmd(self,cmd,script_name=func)

    
    def fsl(self,func):
        cmd = f'export FSLVERSION= {self.exportfsl} && {func}'
        _run_cmd(self,cmd,script_name=func)

    
    #def matlab(input,output,func):
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

    
    

class Stage(BaseStep,Commands):
    process_name = PROCESS_TITLE
    step_name = 'stage'
    step_cli = 'stage'
    cpu = 1
    mem = '1G'


    def __init__(self, project, code, args):
        super().__init__(project, code, args)
        self.next_step = None
        #self.asegstat = 
        

    # def import(self):
    #     fs_root = os.path.join(get_project_dir(project), 'Freesurfer')
    #     raw_mgz = os.path.join(fs_root,'subjects',self.code,'mri','T1.mgz')
    #     new_mgz = os.path.join(self.working_dir,self.code + '.T1.nii.gz')
    #     _run_cmd(self, qit(in,out), script_name='T1_convert_mgz')

    # def run(self):
	#     self.import()
    def aseg_convert(self,aseg_raw): 
        cmd = f'asegstats2table -i {aseg_raw} -d comma -t {self.asegstats}'
        commands.fs(cmd)
    
    def mgz_convert(self,mgz,nii):
        cmd = f'mri_convert {mgz} {nii}'
        commands.fs(cmd)
    
    def make_mask(self,maskmgz):
        img = nib.load(maskmgz)
        data = img.get_fdata()
        mask = np.zeros(np.shape(data))

        seg = [2,10,11,12,13,26,41,49,50,51,52,58]
        wmh = 77    # do we need this if it should already not be included in the others? or is it included? or maybe it doesn't matter since 
                    # she wants me to use SPM anyway

        for m in seg:
            mask[data == m] = m

        maskimg = nib.Nifti1Image(mask,img.affine)
        nib.save(maskimg,self.wmdgmask)

        #return masknii     # i dont think i need this since the variable already exists and i'm just creating it here?
    

    def _parse_args(self, args):
        """
        You may want to parse additional step args.
        """
        parser = argparse.ArgumentParser()
        parser.add_argument("", "", dest="")
        options, _ = parser.parse_known_args(args)
        self.args = "".join(self.args)

        return options

    def call_a_shell_program(self):
        """
        Example of how to lookup the official path of a program from
        the pijp configuration file and call that program.
        """
        # This creates a script which is then executed.
        # Always try to use this method as if self documents the processing
        # and makes debugging easier.
        dcm2niix = util.configuration['dcm2niix']['path']
        cmd = f"{dcm2niix} -z y -b y -o {self.working_dir} -f {self.output}"
        self._run_cmd(cmd, 'dcm2niix')

    @classmethod
    def get_queue(cls, project_name):
        """
        Example how to write a `queue` mode class function that `pijp/engine` will
        call. There are many things you might do here, this is just one simple
        example.
        """
        # Use use the database VIEW `ImageList.<project>` to get the SeriesCodes for the imagetype we need.
        all_codes = ProcessingLog().get_project_images(project_name, image_type=None)

        # We typically want to exclude codes we've already run or attempted.
        attempted_rows = ProcessingLog().get_step_attempted(project_name, PROCESS_TITLE, 'stage')
        attempted = [row[1] for row in attempted_rows]

        # Create a final list
        todo = [{'ProjectName': project_name, "Code": row['Code']} for row in all_codes if row['Code'] not in attempted]

        return todo

    def run(self):
        """
        This is the entry point method is called by `pijp/engine`.
        Start your code here. Create "helper" methods to keep the code clean.
        """
        pass


class Analyze(Stage):
    def __init__(self, project, code, args):
        super().__init__(project, code, args)
        self.next_step = None
        
        # variables specific to this class
        self.count = 0
        self.volume = 0
        self.icv = 0
        self.icv_normed = 0
        self.pvsstats = os.path.join(self.working_dir, self.code+'-pvsstats.csv')


    def frangi_analysis(self,t1,mask,threshold):
        # hessian calculation
        hes =  os.path.join(self.working_dir,self.code+'-hessian.nii.gz')
        cmd_hes = f'VolumeFilterHessian --input {t1} --mask {mask} --mode Norm --output {hes}'
        commands.qit(cmd_hes)

        hes_stats = os.path.join(self.working_dir,self.code+'hessianstats.csv')
        cmd_hesstats = f'VolumeMeasure --input {hes} --output {hes_stats}'
        commands.qit(cmd_hesstats)

        hes_csv = pd.read_csv(hes_stats,index_col=0)
        half_max = hes_csv.loc['max'][0]/2

        # frangi calculation
        frangi_mask = os.path.join(self.working_dir,self.code+'frangimask.nii.gz')
        cmd_frangi = f'VolumeFilterFrangi --input {t1} --mask {mask} --low {0.1} --high {5.0} --scales {10} --gamma {half_max} --dark --output {frangi_mask}'
        commands.qit(cmd_frangi)
        
        # threshold calculation
        cmd_threshold = f'VolumeThreshold --input {frangi_mask} --mask {mask} --threshold {threshold} --output {self.frangimask}'
        commands.qit(cmd_threshold)
    

    def icv_calc(self,aseg_file):
        stat = pd.read_csv(aseg_file)
        self.icv = stat['EstimatedTotalIntrCranialVol'][0]
    
    def pvs_stats(self,frangithreshold):
        """ Calculates pvs stats. Fills in variables count and vol, returns stats table as calculated by MaskMeasure. """
        comp = os.path.join(self.working_dir,self.code+'-frangi_comp.nii.gz')
        cmd_comp = f'Mask Components --input {frangithreshold} --output {comp}'
        commands.qit(cmd_comp)

        cmd_maskmeas = f'MaskMeasure --input {comp} --comps --counts --output {self.pvsstats}'
        commands.qit(cmd_maskmeas)

        stats = pd.read_csv(self.pvsstats,index_col=0)
        self.count =  stats.loc['component_count'][0]    # number of PVS counted
        self.vol = stats.loc['component_sum'][0]       # number of voxels

        self.icv_normed = self.vol / self.icv

        return stats


def run():
    import sys
    current_module = sys.modules[__name__]
    run_module(current_module)


if __name__ == "__main__":
    run_file(os.path.abspath(__file__))



col = ['subjects','research group','pvsvol','pvscount','pvsvolwm','pvscountwm','icv norm','icv norm wm']

alldata = pd.DataFrame(data=zip(subjname_list,researchgroup_list,volpvs_list,countpvs_list,volpvs_list_wm,countpvs_list_wm,icv_norm_list,icv_norm_list_wm),index=np.arange(len(subjname_list))+1,columns=col)

alldata.to_csv(os.path.join(working_dir,'frangidata.csv'), index=True)

