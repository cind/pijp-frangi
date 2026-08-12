"""
Worker script for a single subject's PVS x ROI extraction. Mirrors the
structure of grid_mcpvs_preprocessing.py: takes explicit CLI args, does one
subject's work, and writes that subject's own partial result rather than
touching any shared file -- concurrent grid jobs can't safely write to the
same CSV at once, so aggregation happens in a separate step after the whole
queue finishes (see aggregate_pvsroi_results.py).

Outputs (both written into the subject's own folder):
    <subject>_roiatlas.nii.gz   -- combined ROI atlas
    <subject>_pvsroi.csv        -- this subject's one-row wide result
"""

import os
import re
import sys
import argparse
import pandas as pd

# pvs_roi_extraction.py is expected to live alongside this script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pvs_roi_extraction import (
    build_roi_atlas,
    process_subject,
    get_icv_from_aseg_stats,
    get_icv_from_mask,
)

def find_pvs_mask(subj_dir):
    """
    nnU-Net leaves input-channel files alongside the prediction output,
    e.g. PVS_226_0000.nii.gz / PVS_226_0001.nii.gz (channel suffixes) next
    to the actual prediction PVS_226.nii.gz. Case numbers can themselves be
    4 digits (e.g. PVS_2068.nii.gz), so excluding by suffix digit-length is
    ambiguous -- instead match the prediction filename's structure directly:
    exactly one number between "PVS_" and ".nii.gz", with nothing else.
    """
    prediction_pattern = re.compile(r"^PVS_\d+\.nii\.gz$")
    candidates = [f for f in os.listdir(subj_dir) if prediction_pattern.match(f)]
    if not candidates:
        raise FileNotFoundError(f"No PVS_*.nii.gz prediction found in {subj_dir} (only channel files?)")
    if len(candidates) > 1:
        print(f"WARNING: multiple PVS prediction files found in {subj_dir}, using {sorted(candidates)[0]}")
    return os.path.join(subj_dir, sorted(candidates)[0])


def main():
    parser = argparse.ArgumentParser(description="Extract PVS counts/volumes per ROI for one subject")
    parser.add_argument("--subj_dir", required=True, help="Subject's folder under ADNI3_try2/<DX>/<subject>")
    parser.add_argument("--subject", required=True, help="Subject folder name, e.g. ADNI3_003_S_6264y00_i100539")
    parser.add_argument("--research_group", required=True, help="DX group, e.g. AD/CN/EMCI/...")
    parser.add_argument("--wmparc_path", required=True, help="Path to this subject's wmparc.mgz")
    parser.add_argument("--aseg_stats_path", default=None,
                         help="Path to this subject's aseg.stats (preferred ICV source)")
    parser.add_argument("--icv_mask_path", default=None,
                         help="Fallback ICV source if aseg.stats isn't available")
    args = parser.parse_args()

    print(f"Processing subject: {args.subject} (group={args.research_group})")

    try:
        if not os.path.exists(args.wmparc_path):
            raise FileNotFoundError(f"wmparc.mgz not found: {args.wmparc_path}")

        pvs_path = find_pvs_mask(args.subj_dir)
        print(f"Using PVS mask: {pvs_path}")

        atlas_out = os.path.join(args.subj_dir, f"{args.subject}_roiatlas.nii.gz")
        atlas, affine, header = build_roi_atlas(args.wmparc_path, out_path=atlas_out)
        print(f"ROI atlas written to: {atlas_out}")

        if args.aseg_stats_path and os.path.exists(args.aseg_stats_path):
            icv = get_icv_from_aseg_stats(args.aseg_stats_path)
        elif args.icv_mask_path and os.path.exists(args.icv_mask_path):
            icv = get_icv_from_mask(args.icv_mask_path)
        else:
            raise FileNotFoundError(
                f"No usable ICV source for {args.subject} "
                f"(aseg_stats_path={args.aseg_stats_path}, icv_mask_path={args.icv_mask_path})"
            )

        row = process_subject(args.subject, args.research_group, pvs_path, atlas, icv)

        out_csv = os.path.join(args.subj_dir, f"{args.subject}_pvsroi.csv")
        pd.DataFrame([row]).to_csv(out_csv, index=False)
        print(f"Result written to: {out_csv}")

    except Exception as e:
        # non-zero exit so the pijp Step marks this subject as failed
        print(f"ERROR processing {args.subject}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
