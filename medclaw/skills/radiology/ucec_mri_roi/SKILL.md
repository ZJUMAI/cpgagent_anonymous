# radiology.ucec_mri_roi

Load cached UCEC T2/DWI MRI ROIs produced by `ucec-mri-pipeline`, export axial
review PNGs, and return structured metadata. This skill does **not** run nnUNet;
segmentation and ROI cropping happen offline.

## Input

- `case_id` is required. The skill resolves case data from `MEDCLAW_CASES_ROOT`
  or `examples/cases/{case_id}/`.
- ROI paths follow the default layout implemented in [`workflow.py`](workflow.py):
  - `radiology/roi/{case_id}_T0_t2_roi.nii.gz`
  - `radiology/roi/{case_id}_T0_dwi_roi.nii.gz`
- Resolution order:
  1. `case.yaml` → `data.radiology.t2_roi_uri` / `dwi_roi_uri`
  2. `case_manifest.json` → `files.radiology_roi`
  3. Default layout filenames above and matching files under `radiology/roi/`
- Stale metadata paths are skipped when a valid local cached ROI is present.
- Raw files under `radiology/nifti/` are never labeled as tumor ROI. If only raw
  MRI exists, the failure explicitly requests the offline preprocessing step.
- Optional overrides: `t2_roi_uri`, `dwi_roi_uri`, `modalities`, `timepoint`.

## Output

For each requested modality the skill exports:

- Axial ROI slice PNG
- Context slice PNG (same slice)
- Overlay PNG (mask overlay when a matching tumor mask exists)
- `ucec_mri_roi_metadata.json` with shape, spacing, source paths, and QC summary

This is research benchmark output only and must not be presented as clinical advice.
