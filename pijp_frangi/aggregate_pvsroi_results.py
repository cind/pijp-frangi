"""
Run this once, after every grid job in the pvsroi_extractv1 queue has
finished. Globs every subject's <subject>_pvsroi.csv (written by
grid_pvsroi_extraction.py) and concatenates them into one cohort-level CSV.

Usage:
    python aggregate_pvsroi_results.py
"""

import glob
import os
import pandas as pd

SUBJECTS_ROOT = '/m/Researchers/SerenaT/deeppvs/for_nnunet/ADNI3_try2'
OUT_CSV = os.path.join(SUBJECTS_ROOT, 'grand_PVS_report_roi_results.csv')


def main():
    pattern = os.path.join(SUBJECTS_ROOT, '*', '*', '*_pvsroi.csv')
    partial_files = sorted(glob.glob(pattern))
    print(f"Found {len(partial_files)} partial result files")

    if not partial_files:
        print("Nothing to aggregate -- check SUBJECTS_ROOT and that grid jobs have completed.")
        return

    frames = []
    failed = []
    for f in partial_files:
        try:
            frames.append(pd.read_csv(f))
        except Exception as e:
            failed.append((f, str(e)))

    if failed:
        print(f"WARNING: {len(failed)} partial files failed to read:")
        for f, err in failed:
            print(f"  {f}: {err}")

    full = pd.concat(frames, ignore_index=True)

    # sanity check: flag any duplicate subject IDs (e.g. reruns leaving stale files)
    dupes = full[full.duplicated(subset='subjects', keep=False)]
    if not dupes.empty:
        print(f"WARNING: {dupes['subjects'].nunique()} subject(s) have duplicate rows -- check for stale reruns:")
        print(dupes['subjects'].unique())

    full.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(full)} subjects x {full.shape[1]} columns to {OUT_CSV}")


if __name__ == "__main__":
    main()
