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

    def example(self):
        """
        Contrived example.
        """
        sql = f"SELECT * FROM ImageList.{self.project}"
        cursor = self.db.execute(sql)
        return _trans_to_dict(cursor)
    