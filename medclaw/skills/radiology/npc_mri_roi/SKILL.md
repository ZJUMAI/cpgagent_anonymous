# radiology.npc_mri_roi

Use the ground-truth NPC tumor mask on the original T1C grid to select and
export MRI review images. The skill never reads the legacy cached 96³ ROI.

## Input

- `case_id` is required. The skill resolves case data from
  `MEDCLAW_CASES_ROOT` or `examples/cases/{case_id}/`.
- Default source layout:
  - `radiology/nifti/{case_id}_T0_t1c.nii.gz`
  - `radiology/masks/{case_id}_T0_primary_tumor.nii.gz`
  - `radiology/masks/{case_id}_T0_lymph_node.nii.gz`
- Resolution order:
  1. Explicit `t1c_uri`, `primary_mask_uri`, or `node_mask_uri`
  2. `case.yaml` source/mask declarations
  3. `case_manifest.json` local NIfTI and mask entries
  4. Default filenames above
- Optional inputs: `roi_kinds`, `timepoint`, and `roi_margin_mm` (default
  12 mm). Default `roi_kinds` is `["primary"]`.

The source image and mask must be finite 3D NIfTI volumes with identical shape
and affine. Empty or unregistered masks are rejected instead of producing an
untrusted crop.

## Selection and output

For each requested mask kind, the skill:

1. selects the axial source-image slice with maximum mask area;
2. expands the in-plane mask bounding box by the requested physical margin;
3. exports a cropped ROI slice;
4. exports the complete source-image context slice;
5. exports a full-slice red mask overlay with a yellow crop box; and
6. records source paths, affine, spacing, mask statistics, voxel/world center,
   selected slice and bounding boxes in `npc_mri_roi_metadata.json`.

This is research benchmark output only and must not be presented as clinical
advice.
