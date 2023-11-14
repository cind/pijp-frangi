"""
Most pipelines will require some custom database query.
This template should help you get started. 

ALWAYS provide logging for the query, `LOGGER.debug(sql)`.
"""
from pijp import database
import logging

LOGGER = logging.getLogger(__name__)

def _trans_to_dict(results):
    """
    Create nice list of dicts where the keys are the column names.
    """
    return [dict(zip(row.keys(), row)) for row in results]

class Repository(object):
    def __init__(self, project):
        self.project = project
        self.db = database.Database("iss")

    def get_researchgroup(self,code):
        """
        Gets the research group (AD, MCI, CN) of a subject (helps place the subject into the right research group folder)
        """
        sql = f"SELECT ResearchGroup FROM ImageList.{self.project} where Code = '{code}'"
        cursor = self.db.execute(sql)
        return _trans_to_dict(cursor)
    
    def get_imagetype(self,code,image_type):
        """
        Gets the specified image for a subject (T1 or FLAIR)
        """
        sql = f"""
            SELECT Code 
            FROM ImageList.{self.project} 
            WHERE ScanCode = '{code}'
                AND ImageType = '{image_type}'
            """
        cursor = self.db.execute(sql)
        return _trans_to_dict(cursor)

    