# debugging module
import ipdb

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
        self.project = project
        self.code = code
        self.researchgroup = repo.Repository(self.project).get_researchgroup(self.code)[0]['ResearchGroup']
        self.working_dir = get_case_dir(self.project, self.researchgroup, self.code)

        # only for freesurfer
        self.mrifolder = os.path.join(self.data, self.code,'mri')
        self.statsfolder = os.path.join(self.data, self.code,'stats')

        # all nii converted
        self.t1 = os.path.join(self.working_dir, self.code + "-T1.nii.gz")
        self.allmask = os.path.join(self.working_dir, self.code + "-allmask.nii.gz")
        self.wmmask = os.path.join(self.working_dir, self.code + "-wmmask.nii.gz")
        self.asegstats = os.path.join(self.working_dir, self.code + "-asegstats.csv")
        self.flair = os.path.join(self.working_dir, self.code + "-FLAIR.nii.gz")
        self.wmhmask = os.path.join(self.working_dir, 'ples_lpa_m' + self.code + '_FLAIR.nii')

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
        cmd+= 'export PATH=$JAVA_HOME/bin:$PATH \n'
        cmd+= f'qit {func}'
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
        cmd = f'export MATLAB_VERSION={self.exportmatlab} && matlab -nodesktop -noFigureWindows -nojvm -nosplash -r {_script}'

        proc = subprocess.Popen(cmd, shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        cwd=os.path.dirname(script)
        )

        output, error = proc.communicate()
        LOGGER.error(error)
        LOGGER.info(output)

        if proc.returncode != 0:
            raise Exception('MATLAB FAILURE. Error:\n' + error)


class Stage(BaseStep):
    process_name = PROCESS_TITLE
    step_name = 'Stage'
    step_cli = 'stage'
    cpu = 4
    mem = '4G'

    def __init__(self, project, code, args):
        super(Stage, self).__init__(project, code, args)
        self.next_step = None
        self.commands = Commands(project, code, args)

    def run(self):
        rg = repo.Repository(self.project).get_researchgroup(self.code)
        LOGGER.debug(f"Research Group {rg}")

        flair_check = repo.Repository(self.project).get_imagetype(self.scan_code,'FLAIR')
        LOGGER.debug(f"FLAIR check {flair_check[0]['Code']}")
        if len(rg) == 0:
            raise ProcessingError("No Research Group found.")

        if len(flair_check) == 0:
            raise ProcessingError("No FLAIR found.")

        elif len(flair_check) > 1:
            raise ProcessingError("Found more than 1 FLAIR.")

        t1mgz = os.path.join(self.mrifolder, 'T1.mgz')
        wmparcmgz = os.path.join(self.mrifolder, 'wmparc.mgz')
        maskmgz = os.path.join(self.mrifolder, 'aparc+aseg.mgz')
        asegstats = os.path.join(self.statsfolder, 'aseg.stats')


        proj_root = get_project_dir(self.project)
        flair_raw = os.path.join(proj_root, 'Raw', self.scan_code, flair_check[0]['Code'] + '.FLAIR.nii.gz')
        if not os.path.exists(flair_raw):
            raise ProcessingError("FLAIR nifti is missing from `Raw`")

        self.mgz_convert(t1mgz, self.t1)
        self.mgz_convert(wmparcmgz, self.wmmask)
        self.aseg_convert(asegstats)
        self.make_allmask(maskmgz)
        self.make_wmhmask(self.t1, flair_raw)

    def aseg_convert(self, aseg_raw):
        cmd = f'asegstats2table -i {aseg_raw} -d comma -t {self.asegstats}'
        self.commands.fs(cmd)
        LOGGER.info(self.code + ': aseg2stats done! ')

    def mgz_convert(self, mgz, nii):
        cmd = f'mri_convert {mgz} {nii}'
        self.commands.fs(cmd)
        LOGGER.debug(f"converted output:{nii}")
        LOGGER.info(self.code + ': mgz_convert done! ')

    def make_allmask(self, maskmgz):
        img = nib.load(maskmgz)
        data = img.get_fdata()
        mask = np.zeros(np.shape(data))

        seg = [2, 10, 11, 12, 13, 26, 41, 49, 50, 51, 52, 58]
        wmh = 77
        # Do we need this if it should already not be included in the others?
        # or is it included? or maybe it doesn't matter since She wants me to
        # use SPM anyway

        for m in seg:
            mask[data == m] = m

        maskimg = nib.Nifti1Image(mask, img.affine)
        nib.save(maskimg, self.allmask)

        LOGGER.info(self.code + ': makeall masks done! ')

    def make_wmhmask(self, t1, input_flair):
        """
        WMH removal
        """

        wmhlesion_folder = os.path.join(self.working_dir, 'wmhlesion')
        os.makedirs(os.path.join(self.working_dir, 'wmhlesion'), exist_ok=True)
        shutil.copy(input_flair, wmhlesion_folder)
        flair = glob.glob(os.path.join(wmhlesion_folder, "*FLAIR.nii.gz"))[0]

        # bias correct flair
        flair_bc = os.path.join(wmhlesion_folder, self.code + "_FLAIRbc.nii.gz")
        biascorrect = f'N4BiasFieldCorrection -i {flair} -o {flair_bc}'
        self.commands.ants(biascorrect)

        #register flair
        flair_reg = os.path.join(wmhlesion_folder, self.code + "_FLAIRbcreg.nii.gz")
        # NOTE: You may want to use a differnet cost function b/c these are different modalities.
        # I think the default is correlation coeff.
        register = f'flirt -in {flair_bc} -ref {t1} -out {flair_reg}'
        self.commands.fsl(register)

        # TODO: this needs to be fixed
        ## currently using LPA
        # first unzip the files
        unzipped_flair = os.path.join(wmhlesion_folder, self.code + '_FLAIR.nii')
        with gzip.open(input_flair, 'rb') as f_in:
            with open(unzipped_flair, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        wmh_cmd = f"""
try
    addpath('/opt/mathworks/MatlabToolkits/spm12_r7219');
    addpath('/opt/mathworks/MatlabToolkits/LST/3.0.0');
    spm_jobman('initcfg');
    ps_LST_lpa('{unzipped_flair}');
catch ME
    report = ME.getReport;
    fprintf(2, report);
    exit(-1);
end
exit;"""

        matlab_script = self._prep_cmd_script(wmh_cmd, 'wmh.m')
        LOGGER.debug(f"MATLAB m file:{matlab_script}")
        self.commands.matlab(matlab_script)

        # Orig file path.
        #wmhmask = os.path.join(wmhlesion_folder, 'ples_lpa_mr' + self.code + '_FLAIR.nii')
        # One that was generated.
        wmhmask = os.path.join(wmhlesion_folder, 'ples_lpa_m' + self.code + '_FLAIR.nii')
        shutil.copy(wmhmask, self.working_dir)

        LOGGER.info(self.code + ': wmhmasks done!')

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


class Analyze(Stage):
    process_name = PROCESS_TITLE
    step_name = 'Analyze'
    step_cli = 'analyze'
    cpu = 4
    mem = '4G'

    def __init__(self, project, code, args=None):
        super().__init__(project, code, args)
        self.next_step = None
        self.commands = Commands(project, code, args)

        # variables specific to this class
        self.count = -1
        self.vol = -1
        self.icv = -1
        self.icv_normed = -1

        self.countwm = -1
        self.volwm = -1
        self.icv_normedwm = -1

        self.pvsstats = os.path.join(self.working_dir, self.code + '-pvsstats.csv')
        self.comp = os.path.join(self.working_dir, self.code + '-frangi_comp.nii.gz')

    def run(self):

        frangimask_all = os.path.join(self.working_dir, self.code + "-frangi-thresholded-wmhrem.nii.gz")
        self.frangi_analysis(self.t1, self.allmask, 0.0025, frangimask_all, wmhmask = self.wmhmask)

        self.icv_calc(self.asegstats)
        count_all, vol_all, icv_all = self.pvs_stats(frangimask_all)

        frangimask_wm = os.path.join(self.working_dir, self.code + "-frangi-thresholded-wm-wmhrem.nii.gz")
        self.frangi_analysis(self.t1, self.wmmask, 0.0002, frangimask_wm, region = 'wm',wmhmask = self.wmhmask)

        self.pvs_stats(frangimask_wm)
        count_allwm, vol_allwm, icv_allwm = self.pvs_stats(frangimask_wm)

        subject = self.code
        researchgroup = self.researchgroup


        #ipdb.set_trace()

        col = ['subjects','research group','pvscount','pvsvol','icv norm','pvscountwm','pvsvolwm','icv norm wm']
<<<<<<< Updated upstream
        df = pd.DataFrame(data=zip(subject, researchgroup, count_all, vol_all, icv_all, count_allwm, vol_allwm, icv_allwm),columns=col)
        df.to_csv(self.working_dir, index=True)
=======
        df = pd.DataFrame(data=[[subject,researchgroup,str(count_all),str(vol_all),str(icv_all),str(count_allwm),str(vol_allwm),str(icv_allwm)]],columns=col)
        df.to_csv(os.path.join(self.working_dir,self.code+'_pvs_info.csv'), index=True)
>>>>>>> Stashed changes

    def frangi_analysis(self, t1, mask, threshold, output, region='all',wmhmask=None):

        # hessian calculation
        hes =  os.path.join(self.working_dir, self.code + '-hessian-' + region + '.nii.gz')
        cmd_hes = f'VolumeFilterHessian --input {t1} --mask {mask} --mode Norm --output {hes}'
        #ipdb.set_trace()

        self.commands.qit(cmd_hes)

        hes_stats = os.path.join(self.working_dir, self.code + '-hessianstats' + region + '.csv')
        cmd_hesstats = f'VolumeMeasure --input {hes} --output {hes_stats}'
        self.commands.qit(cmd_hesstats)

        hes_csv = pd.read_csv(hes_stats, index_col=0)
        half_max = hes_csv.loc['max'][0]/2

        # frangi calculation
        frangi_mask = os.path.join(self.working_dir, self.code +'-frangimask' + region + '.nii.gz')
        cmd_frangi = f'VolumeFilterFrangi --input {t1} --mask {mask} --low {0.1} --high {5.0} --scales {10} --gamma {half_max} --dark --output {frangi_mask}'
        self.commands.qit(cmd_frangi)

        if wmhmask is not None:
            pre_output = os.path.join(self.working_dir, self.code + '-frangimask' + region + '-thresholded.nii.gz')
            cmd_threshold = f'VolumeThreshold --input {frangi_mask} --mask {mask} --threshold {threshold} --output {pre_output}'
            self.commands.qit(cmd_threshold)

            cmd_removewmh = f'MaskSet --input {pre_output} --mask {wmhmask} --label {0} --output {output}'
            self.commands.qit(cmd_removewmh)

        else:
            cmd_threshold = f'VolumeThreshold --input {frangi_mask} --mask {mask} --threshold {threshold} --output {output}'
            self.commands.qit(cmd_threshold)

        LOGGER.info(self.code + ': frangi analysis done! ')

    def icv_calc(self, asegstats):
        stat = pd.read_csv(asegstats)
        self.icv = stat['EstimatedTotalIntraCranialVol'][0]

        LOGGER.info(self.code + ': icv calc done! ')


    def pvs_stats(self, frangimask):
        """ Calculates pvs stats. Fills in variables count and vol, returns stats table as calculated by MaskMeasure. """
        cmd_comp = f'MaskComponents --input {frangimask} --output {self.comp}'
        self.commands.qit(cmd_comp)

        cmd_maskmeas = f'MaskMeasure --input {self.comp} --comps --counts --output {self.pvsstats}'
        self.commands.qit(cmd_maskmeas)

        stats = pd.read_csv(self.pvsstats, index_col=0)
        count =  stats.loc['component_count'][0]    # number of PVS counted
        vol = stats.loc['component_sum'][0]       # number of voxels

        self.vol = vol
        self.count = count

        icv_normed = vol / self.icv

        self.icv_normed = icv_normed




        LOGGER.info(self.code + ': pvs stats done! ')

        return count, vol, icv_normed


def run():
    import sys
    current_module = sys.modules[__name__]
    run_module(current_module)


if __name__ == "__main__":
    run_file(os.path.abspath(__file__))


