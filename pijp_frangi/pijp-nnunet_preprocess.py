import os
import logging
import argparse
import datetime
import subprocess

from pijp import util
from pijp.repositories import ProcessingLog
from pijp.core import Step, get_project_dir
from pijp.engine import run_module, run_file
from pijp.exceptions import ProcessingError

LOGGER = logging.getLogger(__name__)
PROCESS_TITLE = 'nnunet_preprocess'

def get_process_dir(project):
    return os.path.join(get_project_dir(project), PROCESS_TITLE)

def get_case_dir(project, research_group, subject):
    """Create directory organized by research group and subject"""
    cdir = os.path.join(get_process_dir(project), research_group, subject)
    if not os.path.isdir(cdir):
        os.makedirs(cdir)
    return cdir


class PreprocessSubject(Step):
    process_name = PROCESS_TITLE
    step_name = 'preprocess'
    step_cli = 'preprocess'
    cpu = 1
    mem = '8G'
    
    def __init__(self, project, code, args):
        super().__init__(project, code, args)
        self.datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
        self.project = project
    
        LOGGER.info(f"Received code: {code}")

        # Parse the research group and subject from the code path
        # code format: /path/to/ADNI3_preprocessed/EMCI/subject_001
        parts = code.rstrip('/').split('/')

            # Debug: print parts
        LOGGER.info(f"Path parts: {parts}")
        LOGGER.info(f"Number of parts: {len(parts)}")

        self.research_group = parts[-2]  # e.g., 'EMCI'
        self.subject = parts[-1]  # e.g., 'subject_001'
        
        LOGGER.info(f"Extracted research_group: {self.research_group}, subject: {self.subject}")

        # Use research_group_subject format for the code to avoid "/" in job names
        self.code = f"{self.research_group}_{self.subject}"
        
        self.subj_dir = code
        self.working_dir = get_case_dir(self.project, self.research_group, self.subject)
        
        self.output_folder = os.path.join(
            '/m/Researchers/SerenaT/deeppvs/for_nnunet/ADNI3_preprocessed_clean',
            self.research_group,
            self.subject
        )
    
    @classmethod
    def get_queue(cls, project_name):
        """
        Build the queue of all subjects to process.
        Returns a list of dicts with ProjectName and Code (subject path).
        """
        parent_dir = '/m/Researchers/SerenaT/deeppvs/for_nnunet/ADNI3_preprocessed'
        dx_names = ['EMCI', 'AD', 'MCI', 'CN', 'LMCI', 'SMC']
        
        # Get already attempted subjects
        attempted_rows = ProcessingLog().get_step_attempted(project_name, PROCESS_TITLE, 'preprocess')
        attempted = [row[1] for row in attempted_rows]
        
        todo = []
        for research_group in dx_names:
            dx_dir = os.path.join(parent_dir, research_group)
            if not os.path.isdir(dx_dir):
                LOGGER.warning(f"Directory not found: {dx_dir}")
                continue
            
            # Get all subject folders
            subjects = [s for s in os.listdir(dx_dir) 
                       if not s.startswith('.') and os.path.isdir(os.path.join(dx_dir, s))]
            
            for subject in subjects:
                subj_path = os.path.join(dx_dir, subject)
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
        LOGGER.info(f"Processing subject: {self.subject} from group: {self.research_group}")
        LOGGER.info(f"Subject directory: {self.subj_dir}")
        LOGGER.info(f"Output folder: {self.output_folder}")
        
        # Create output directory
        os.makedirs(self.output_folder, exist_ok=True)
        
        # Get the full path to the preprocessing script
        # Assumes it's in the same directory as this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        preprocess_script = os.path.join(script_dir, 'grid_nnunet_preprocessing.py')
        
        # Run your preprocessing Python script
        cmd = [
            'python',
            preprocess_script,
            '--subj_dir', self.subj_dir,
            '--subject', self.subject,
            '--output_folder', self.output_folder
        ]
        
        LOGGER.info(f"Running command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding='utf-8',
                check=True,
                cwd=script_dir  # Run from script directory
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


def run():
    import sys
    current_module = sys.modules[__name__]
    run_module(current_module)


if __name__ == "__main__":
    run_file(os.path.abspath(__file__))