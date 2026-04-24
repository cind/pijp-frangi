from pijp.repositories import ProcessingLog

# first, just view what's in the log for your project
log = ProcessingLog()
rows = log.get_step_attempted('ADNI3_frangi', 'mcpvs_preprocess', 'preprocess')
print(f"Found {len(rows)} logged subjects")
print(rows[:3])  # preview first few

print([m for m in dir(log) if not m.startswith('_')])
