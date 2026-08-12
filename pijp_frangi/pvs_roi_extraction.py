"""
Per-subject, per-ROI PVS cluster count/volume extraction, with ROI volumes
and intracranial volume (ICV) for later normalization.

Pipeline:
  1. build_roi_atlas(): collapse wmparc.mgz sub-segmentations into a single
     combined ROI atlas per subject, one clean integer ID per ROI (lobes,
     BG structures, hippocampus, VDC, brainstem -- hemisphere-specific).
  2. process_subject(): label the whole-brain PVS mask ONCE (cc3d,
     26-connectivity), then assign each cluster to a single ROI via a
     sparse (cluster x ROI) voxel-overlap crosstab, resolved by argmax
     (majority overlap). Also computes each ROI's own volume, independent
     of PVS.
  3. get_icv(): pulls eTIV from a FreeSurfer aseg.stats file (recommended),
     or falls back to counting voxels in an explicit ICV/brain mask.
  4. run_cohort(): loops subjects sequentially, writes one combined CSV.

Install: pip install connected-components-3d nibabel pandas scipy
"""

import re
import numpy as np
import nibabel as nib
import pandas as pd
from scipy import sparse
import cc3d
from pathlib import Path


# ---------------------------------------------------------------------------
# 1. ROI atlas construction from wmparc.mgz
# ---------------------------------------------------------------------------

# name -> list of wmparc/aseg segmentation IDs that belong to that ROI.
# NOTE: rh_temporal corrected from the original 4040 -> 4030 to mirror
# lh_temporal's 3030. Double check this against the FreeSurfer LUT for
# your specific atlas version before trusting results.
ROI_DEFINITIONS = {
    "lh_frontal":     [3003, 3012, 3017, 3014, 3018, 3019, 3020, 3024, 3027, 3028, 3032],
    "rh_frontal":     [4003, 4012, 4017, 4014, 4018, 4019, 4020, 4024, 4027, 4028, 4032],
    "lh_parietal":    [3008, 3022, 3025, 3029, 3031],
    "rh_parietal":    [4008, 4022, 4025, 4029, 4031],
    "lh_temporal":    [3001, 3006, 3007, 3009, 3015, 3030, 3033, 3034, 3016],
    "rh_temporal":    [4001, 4006, 4007, 4009, 4015, 4030, 4033, 4034, 4016],
    "lh_occipital":   [3005, 3011, 3013, 3021],
    "rh_occipital":   [4005, 4011, 4013, 4021],
    "lh_cingulate":   [3002, 3026, 3023, 3010],
    "rh_cingulate":   [4002, 4026, 4023, 4010],
    "lh_thalamus":    [10],
    "rh_thalamus":    [49],
    "lh_striatum":    [11, 12, 26],
    "rh_striatum":    [50, 51, 58],
    "lh_pallidum":    [13],
    "rh_pallidum":    [52],
    "lh_hippocampus": [17],
    "rh_hippocampus": [53],
    "lh_vdc":         [28],
    "rh_vdc":         [60],
    "brainstem":      [16],
}

# fixed ROI name -> integer ID map, built once so it's identical across subjects
ROI_NAME_TO_ID = {name: i + 1 for i, name in enumerate(ROI_DEFINITIONS)}
ROI_ID_TO_NAME = {v: k for k, v in ROI_NAME_TO_ID.items()}


def build_roi_atlas(wmparc_path, out_path=None):
    """
    Collapse wmparc.mgz into a single combined ROI atlas where every voxel
    belonging to a given ROI gets that ROI's integer ID (per ROI_NAME_TO_ID),
    rather than the original FreeSurfer sub-segmentation value.

    Returns (atlas_array, affine, header). Optionally saves to out_path.
    """
    img = nib.load(wmparc_path)
    data = np.asarray(img.dataobj)

    atlas = np.zeros(data.shape, dtype=np.int16)
    for name, seg_ids in ROI_DEFINITIONS.items():
        roi_id = ROI_NAME_TO_ID[name]
        atlas[np.isin(data, seg_ids)] = roi_id

    if out_path is not None:
        nib.save(nib.Nifti1Image(atlas, img.affine, img.header), out_path)

    return atlas, img.affine, img.header


# ---------------------------------------------------------------------------
# 2. Intracranial volume
# ---------------------------------------------------------------------------

def get_icv_from_aseg_stats(stats_path):
    """
    Parse eTIV (Estimated Total Intracranial Volume) out of a FreeSurfer
    aseg.stats file. This is the standard/recommended source for ICV --
    prefer this over deriving it from a mask when a stats file is available.
    """
    text = Path(stats_path).read_text()
    match = re.search(r"EstimatedTotalIntraCranialVol.*?,\s*([\d.]+)\s*,\s*mm", text)
    if not match:
        raise ValueError(f"Could not find eTIV in {stats_path}")
    return float(match.group(1))


def get_icv_from_mask(mask_path):
    """
    Fallback: compute ICV by counting nonzero voxels in an explicit
    intracranial/brain mask (e.g. brainmask.mgz > 0) x voxel volume.
    Use only if an aseg.stats file isn't available.
    """
    img = nib.load(mask_path)
    data = np.asarray(img.dataobj)
    voxel_vol = float(np.prod(img.header.get_zooms()[:3]))
    return float(np.count_nonzero(data)) * voxel_vol


# ---------------------------------------------------------------------------
# 3. Subject ID / research group parsing from path
# ---------------------------------------------------------------------------

ADNI_ID_PATTERN = re.compile(r"\d{3}_S_\d{4}")


def parse_subject_path(path):
    """
    Extract subject_id and research_group from a subject-level path.

    ADJUST THIS to match your actual directory layout -- ADNI's diagnostic
    group (CN/MCI/AD/EMCI/LMCI/SMC) is usually NOT part of the raw ADNI file
    path and typically requires a lookup against ADNIMERGE/DXSUM instead.
    This assumes your local layout encodes it, e.g.:
        .../ADNI3/CN/002_S_0295/wmparc.mgz
        .../ADNI3/AD/002_S_1234/wmparc.mgz
    i.e. group is the parent directory two levels above the subject folder.
    Change the indexing below (or replace with a merge against a group
    lookup table) to match your real structure -- check it against one
    known path before running the full cohort.
    """
    parts = Path(path).parts
    id_match = ADNI_ID_PATTERN.search(path)
    subject_id = id_match.group(0) if id_match else None

    research_group = None
    if subject_id in parts:
        idx = parts.index(subject_id)
        if idx > 0:
            research_group = parts[idx - 1]

    return subject_id, research_group


# ---------------------------------------------------------------------------
# 4. Per-subject PVS x ROI extraction (wide format: one row per subject)
# ---------------------------------------------------------------------------

def process_subject(subject_id, research_group, pvs_path, roi_atlas, icv_mm3,
                     roi_id_to_name=ROI_ID_TO_NAME, connectivity=26):
    """
    subject_id / research_group: pass these explicitly rather than re-parsing
    a path here -- the caller (e.g. a pijp Step, or parse_subject_path for
    the standalone cohort runner below) already knows them cleanly.
    roi_atlas: array from build_roi_atlas
    icv_mm3: this subject's intracranial volume

    Returns a single-row dict:
        subjects, research group,
        pvscount_<roi>_<id>, pvsvol_<roi>_<id>, <roi>_<id>VOL  (per ROI),
        icv
    """
    pvs_img = nib.load(pvs_path)
    pvs = np.asarray(pvs_img.dataobj).astype(bool)

    if pvs.shape != roi_atlas.shape:
        raise ValueError(
            f"[{subject_id}] shape mismatch: PVS {pvs.shape} vs ROI atlas {roi_atlas.shape}. "
            "Resample to the same space before running this."
        )

    voxel_vol = float(np.prod(pvs_img.header.get_zooms()[:3]))
    all_roi_ids = np.array(sorted(roi_id_to_name.keys()))

    # ROI volumes are independent of PVS
    roi_flat_full = roi_atlas.ravel()
    roi_voxel_counts = np.bincount(roi_flat_full, minlength=int(roi_flat_full.max()) + 1)
    roi_volume_lookup = {rid: roi_voxel_counts[rid] * voxel_vol if rid < len(roi_voxel_counts) else 0.0
                          for rid in all_roi_ids}

    # per-ROI PVS count/volume, default to 0 (overwritten below if PVS clusters exist)
    pvs_count_lookup = {rid: 0 for rid in all_roi_ids}
    pvs_volume_lookup = {rid: 0.0 for rid in all_roi_ids}

    labeled = cc3d.connected_components(pvs, connectivity=connectivity)
    n_components = int(labeled.max())

    if n_components > 0:
        labeled_flat = labeled.ravel()
        fg = labeled_flat > 0
        lab_fg = labeled_flat[fg]
        roi_fg = roi_flat_full[fg]
        max_roi = int(roi_flat_full.max())

        crosstab = sparse.coo_matrix(
            (np.ones(lab_fg.size, dtype=np.int32), (lab_fg, roi_fg)),
            shape=(n_components + 1, max_roi + 1),
        ).tocsr()
        crosstab[:, 0] = 0  # never assign a cluster to background

        assigned_roi = np.asarray(crosstab.argmax(axis=1)).ravel()
        comp_sizes = np.bincount(lab_fg, minlength=n_components + 1)

        comp_df = pd.DataFrame({
            "component": np.arange(1, n_components + 1),
            "roi": assigned_roi[1:],
            "voxels": comp_sizes[1:],
        })
        comp_df = comp_df[comp_df["roi"] != 0]

        grouped = comp_df.groupby("roi").agg(pvs_count=("component", "size"),
                                              pvs_voxels=("voxels", "sum"))
        for rid, row in grouped.iterrows():
            pvs_count_lookup[rid] = int(row["pvs_count"])
            pvs_volume_lookup[rid] = float(row["pvs_voxels"]) * voxel_vol

    # --- assemble the wide row ---
    row = {"subjects": subject_id, "research group": research_group}
    for rid in all_roi_ids:
        roi_tag = f"{roi_id_to_name[rid]}_{rid}"
        row[f"pvscount_{roi_tag}"] = pvs_count_lookup[rid]
        row[f"pvsvol_{roi_tag}"] = pvs_volume_lookup[rid]
        row[f"{roi_tag}VOL"] = roi_volume_lookup[rid]
    row["icv"] = icv_mm3

    return row


# ---------------------------------------------------------------------------
# 5. Cohort runner (sequential -- grid-job version comes next)
# ---------------------------------------------------------------------------

def run_cohort(subject_table, out_csv="pvs_roi_results.csv"):
    """
    subject_table: DataFrame with columns:
        pvs_path, wmparc_path, and EITHER aseg_stats_path OR icv_mask_path
        (whichever ICV source you're using). subject_id/research_group are
        parsed from wmparc_path via parse_subject_path -- no separate
        subject_id column needed unless your paths don't encode it cleanly.

    Writes one row per subject, wide format, to out_csv.
    """
    rows = []
    for row in subject_table.itertuples(index=False):
        atlas, affine, header = build_roi_atlas(row.wmparc_path)
        subject_id, research_group = parse_subject_path(row.wmparc_path)

        if hasattr(row, "aseg_stats_path") and pd.notna(row.aseg_stats_path):
            icv = get_icv_from_aseg_stats(row.aseg_stats_path)
        else:
            icv = get_icv_from_mask(row.icv_mask_path)

        rows.append(process_subject(subject_id, research_group, row.pvs_path, atlas, icv))

    full = pd.DataFrame(rows)
    full.to_csv(out_csv, index=False)
    return full


if __name__ == "__main__":
    subjects = pd.DataFrame({
        "pvs_path": ["/path/to/ADNI3/CN/002_S_0295/pvs_mask.nii.gz",
                     "/path/to/ADNI3/AD/002_S_0413/pvs_mask.nii.gz"],
        "wmparc_path": ["/path/to/ADNI3/CN/002_S_0295/wmparc.mgz",
                        "/path/to/ADNI3/AD/002_S_0413/wmparc.mgz"],
        "aseg_stats_path": ["/path/to/ADNI3/CN/002_S_0295/aseg.stats",
                            "/path/to/ADNI3/AD/002_S_0413/aseg.stats"],
    })

    df = run_cohort(subjects, out_csv=str(Path.cwd() / "pvs_roi_results.csv"))
    print(df.head())
