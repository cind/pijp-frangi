import os
import subprocess
import logging
import argparse
import textwrap
import string
import datetime

import numpy as np
import shutil
import pandas as pd
import sys
import gzip
import nibabel as nib
from matplotlib import pyplot as plt
from scipy import stats
import glob

# Before loading NUMPY: HACK to avoid seg fault on some hosts.
os.environ["OMP_NUM_THREADS"] = "1"

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
        super(BaseStep, self).__init__(project, code, args)
        self.datetime = datetime.datetime.now().strftime('%Y-%m-%d-%H%M%s')
        #  ADNI3_frangi gets its FS data from ADNI3_FSdn (Denoise).
        self.data = os.path.join(get_project_dir('ADNI3_FSdn'),'Freesurfer','subjects')
        self.proj_root = get_project_dir(self.project)
        self.project = project
        self.code = code
        self.researchgroup = repo.Repository(self.project).get_researchgroup(self.code)[0]['ResearchGroup']
        self.working_dir = get_case_dir(self.project, self.researchgroup, self.code)
        # only for freesurfer
        self.mrifolder = os.path.join(self.data, self.code,'mri')
        self.statsfolder = os.path.join(self.data, self.code,'stats')
        # all nii
        self.t1 = os.path.join(self.working_dir, self.code + "-T1.nii.gz")
        self.allmask = os.path.join(self.working_dir, self.code + "-allmask.nii.gz")
        self.wmmask = os.path.join(self.working_dir, self.code + "-wmmask.nii.gz")
        self.asegstats = os.path.join(self.working_dir, self.code + "-asegstats.csv")
        self.flair = os.path.join(self.working_dir, self.code + "-FLAIR.nii.gz")
        self.wmhmask = os.path.join(self.working_dir, self.code + '-wmhmask.nii')    
        self.greymask = os.path.join(self.working_dir, self.code + "-gmmask.nii.gz")
        self.wmhmask2 = os.path.join(self.working_dir, self.code + '-wmhmask_thresh.nii')
        self.total_wmhmask = os.path.join(self.working_dir, self.code + '-wmhmask_total.nii')

        self.t1raw = os.path.join(self.working_dir, self.code + "-T1raw.nii.gz")
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
    # commands.fs(func) where func = f'some command -i {input} -o {output},
    # then input and output are specified at the actual functions --> at the
    # moment, I have it so this already includes running the command

    def __init__(self, project, code, args=None):
        super(Commands, self).__init__(project, code, args)
        self.exportfs = '7.3.2'
        self.exportants = 'ants-2017-12-07'
        self.exportfsl = '6.0.0'
        self.exportmatlab = 'R2019a'

    def qit(self, func):
        cmd = 'export JAVA_HOME=/opt/qit/jdk-12.0.2 \n'
        cmd += 'export PATH=$JAVA_HOME/bin:$PATH \n'
        cmd += 'export _JAVA_OPTIONS="-Xmx4G -XX:ActiveProcessorCount=1"\n'
        cmd += f'qit {func}'
        self._run_cmd(cmd, script_name='qit_func')     # do i need self in these?
        # script_name='qitfunc'

    def fs(self, func):
        cmd = f'export FSVERSION={self.exportfs} && {func}'
        self._run_cmd(cmd, script_name='fs_func')

    def ants(self, func):
        ncores = Stage.cpu
        # We must use the same number of cores as the class has set.
        # We can increase, this but then we limit the number of overall parallel jobs.
        cmd = f'export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS={ncores} && export ANTSVERSION={self.exportants} && {func}'
        self._run_cmd(cmd, script_name='ants_func')

    def fsl(self, func):
        cmd = f'export FSLVERSION={self.exportfsl} && {func}'
        self._run_cmd(cmd,
                script_name='fsl_func')

    def matlab(self, script):
        #  We need the sub process, to CD to the directory with the m file.
        # Then strip off the `.m` extension.
        _script = os.path.basename(script).replace(".m", "")
        cmd = f'export MATLAB_VERSION={self.exportmatlab} && matlab -singleCompThread -nodesktop -noFigureWindows -nojvm -nosplash -r {_script}'

        proc = subprocess.Popen(cmd, shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        cwd=os.path.dirname(script)
        )

        output, error = proc.communicate()
        LOGGER.error(error)
        LOGGER.info(output)

        if proc.returncode != 0:
            error = error.decode('ascii', errors='ignore')
            raise Exception('MATLAB FAILURE. Error:\n' + error)

class Stage(BaseStep):
    """
    Collect images, and run LST in MATLAB.
    LST uses SPM and requires more than 4GB of memory.
    """
    process_name = PROCESS_TITLE
    step_name = 'Stage'
    step_cli = 'stage'
    cpu = 2
    mem = '16G'

    def __init__(self, project, code, args):
        super(Stage, self).__init__(project, code, args)
        self.next_step = Analyze
        self.commands = Commands(project, code, args)

    def run(self):
        wmparcmgz = os.path.join(self.mrifolder, 'wmparc.mgz')
        asegstats = os.path.join(self.working_dir, self.code+'-asegstats.csv')
        
        # for report
        subject = self.code
        researchgroup = self.researchgroup
        self.icv_calc(self.asegstats)

        if os.path.exists(self.wmhmask):
            frangimask_all = os.path.join(self.working_dir, self.code + "-frangi-thresholded-wmhrem.nii.gz")
            self.frangi_analysis(self.t1, self.allmask, 0.0002, frangimask_all, wmhmask = self.wmhmask)
            count_all, vol_all, icv_all = self.pvs_stats(frangimask_all,self.comp,self.pvsstats)

            frangimask_wm = os.path.join(self.working_dir, self.code + "-frangi-thresholded-wm-wmhrem.nii.gz")
            self.frangi_analysis(self.t1, self.wmmask, 0.0002, frangimask_wm, region = 'wm',wmhmask = self.wmhmask)
            count_allwm, vol_allwm, icv_allwm = self.pvs_stats(frangimask_wm,self.comp_wm,self.pvsstats_wm)
            
            raw = 'no'
            WMHstatus = 'yes'

        else:
            frangimask_all = os.path.join(self.working_dir, self.code + "-frangi-thresholded-wmhrem.nii.gz")
            self.frangi_analysis(self.t1, self.allmask, 0.0002, frangimask_all)
            count_all, vol_all, icv_all = self.pvs_stats(frangimask_all,self.comp,self.pvsstats)

            frangimask_wm = os.path.join(self.working_dir, self.code + "-frangi-thresholded-wm-wmhrem.nii.gz")
            self.frangi_analysis(self.t1, self.wmmask, 0.0002, frangimask_wm, region = 'wm')
            count_allwm, vol_allwm, icv_allwm = self.pvs_stats(frangimask_wm,self.comp_wm,self.pvsstats_wm)

            raw = 'no'
            WMHstatus = 'no'
        col = ['subjects','icv','wmhvol','icv norm wmhvol','wmvol','bgvol','hippvol','raw','WMH mask']
        df_empty = pd.DataFrame(columns=col)
        datatable = os.path.join(self.proj_root,'grand_PVS_report.csv')
        if os.path.exists(datatable):
            df_data = pd.read_csv(datatable)
            newsubject = pd.DataFrame(data=[[subject, researchgroup, count_all, vol_all, icv_all, count_allwm, vol_allwm, icv_allwm, raw, WMHstatus]],columns=col)
            df_data = df_data.append(newsubject)
        else:
            df_empty.to_csv(datatable,index=False)
            
            df_data = pd.read_csv(datatable)
            newsubject = pd.DataFrame(data=[[subject, researchgroup, count_all, vol_all, icv_all, count_allwm, vol_allwm, icv_allwm, raw, WMHstatus]],columns=col)
            df_data = df_data.append(newsubject)
        
        # clean any duplicates
        df_cleaned = df_data
        df_cleaned.drop_duplicates(subset='subjects',keep='last',inplace=True)
        df_cleaned.to_csv(datatable,index=False)

        # for individual report
        newsubject.to_csv(os.path.join(self.working_dir, self.code+'_report.csv'), index=False)

#########----------------------------------WORKING ON FUNCTIONS HERE------------------------------------------------##################
    # WMH stats function: done
    def mask_stats(self,mask,statsname):
    
        cmd_maskmeas = f'MaskMeasure --input {mask} --output {statsname}'
        self.commands.qit(cmd_maskmeas)

        stats = pd.read_csv(statsname, index_col=0)
        vol = stats.loc['volume'][0]       # number of voxels

        #just for the purposes of logging, self.xxx variables not actually used
        self.vol = vol
        icv_normed = vol / self.icv
        
        return vol, icv_normed

    def icv_calc(self, asegstats):
        stat = pd.read_csv(asegstats)
        self.icv = stat['EstimatedTotalIntraCranialVol'][0]

        LOGGER.info(self.code + ': icv calc done! ')

    # make mask
    def make_mask(self,maskmgz,seg,maskname):
        img = nib.load(maskmgz)
        data = img.get_fdata()
        mask = np.zeros(np.shape(data))

        # seg = [10, 11, 12, 13, 26, 49, 50, 51, 52, 58]
        # wmh = 77
        # wm = [2,41] # took these out 12/4/23
        # Do we need this if it should already not be included in the others?
        # or is it included? or maybe it doesn't matter since She wants me to
        # use SPM anyway

        for m in seg:
            mask[data == m] = m

        maskimg = nib.Nifti1Image(mask, img.affine)
        nib.save(maskimg, maskname)


    # obtain BG, HIPP segmentations
    # either with all new threshold or do the intersection method with qit
    # measure and put everything into the new subject table (above)


    # obtain WM, BG, HIPP volumes
    # figure out where these values are in the stats table and put them into new subject table (above)
            


    # ASL measures
    # not here yet


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
        # Use use the database VIEW `ImageList.<project>` to get the SeriesCodes for the `Step` we need.
        all_codes = ProcessingLog().get_project_images(project_name, image_type='T1')
        #LOGGER.info('from queue method: all_codes = ' + all_codes)
        #print(all_codes)

        # We typically want to exclude codes we've already run or attempted.
        attempted_rows = ProcessingLog().get_step_attempted(project_name, PROCESS_TITLE, 'stage')
        #LOGGER.info('from queue method: attempted_rows = ' + attempted_rows)
        #print(attempted_rows)

        attempted = [row[1] for row in attempted_rows]
        #LOGGER.info('from queue method: attempted = ' + attempted)
        #print(attempted)

        # Create a final list
        todo = [{'ProjectName': project_name, "Code": row['Code']} for row in all_codes if row['Code'] not in attempted]
        #LOGGER.info('from queue method: todo = ' + todo)
        #print(todo)

        return todo


def run():
    import sys
    current_module = sys.modules[__name__]
    run_module(current_module)


if __name__ == "__main__":
    run_file(os.path.abspath(__file__))