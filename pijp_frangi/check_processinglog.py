from pijp.repositories import ProcessingLog

# first, just view what's in the log for your project
log = ProcessingLog()
rows = log.get_step_attempted('ADNI3_frangi', 'mcpvs_preprocess', 'preprocess')
print(f"Found {len(rows)} logged subjects")
print(rows[:3])  # preview first few

print([m for m in dir(log) if not m.startswith('_')])

from pijp.core import get_project_dir
print(get_project_dir('ADNI3_frangi'))

from pijp.repositories import ProcessingLog
log = ProcessingLog()
print(log.db)
print(log.connection)

print(dir(log.db))
print(log.db.__dict__)

import pijp
print(pijp.__file__)  # shows where pijp is installed
print(pijp.__dict__)  # shows pijp's attributes

# look at pijp config
from pijp import config
print(config.__file__)
print(config.__dict__)