# radiology.lung_tumor_roi

Segments a lung tumor from a preprocessed CT volume, keeps the largest
three-dimensional connected tumor component, selects the axial slice where that
component has the largest area, and exports a two-dimensional ROI with context.

## Input

- `case_id` is required. The skill first honors `data.ct_preprocessed_uri` in
  the case's `case.yaml` when present. Otherwise it looks for exactly one
  `*_ct_preprocessed.nii.gz` file, preferring
  `radiology/nifti/*_ct_preprocessed.nii.gz` under the matching case directory.
  The case root is `MEDCLAW_CASES_ROOT` when set, otherwise the repository
  `examples/cases/` directory.
- `ct_uri` is optional and may be an absolute local path, repository-relative
  path, or `file://` URI. It is intended for trusted advanced callers; agents
  should normally omit it.
- `tumor_mask_uri` is an optional trusted override for a cached mask aligned to
  the preprocessed CT. A standard sibling
  `radiology/masks/{case_id}_T0_tumor_preprocessed.nii.gz` is discovered
  automatically. When a cached mask is available, model inference and GPU use
  are skipped.

The CT must be a three-dimensional, approximately 1 mm isotropic, lung-window
normalized NIfTI volume with finite values near the `[0, 1]` range.
Raw CT NIfTIs are reported as preprocessing inputs but are not silently treated
as normalized model inputs.

The fixed inference configuration is LungTumorMask 1.3.1 with lung filtering,
threshold `0.5`, morphology radius `3`, and a `32 px` ROI margin. Device
selection is controlled by `MEDCLAW_LUNG_TUMOR_DEVICE=cuda|auto|cpu`; the
default is `cuda`.
The skill applies a narrow compatibility shim for LungTumorMask's removed MONAI
`AddChanneld`, `alias`, and `export` imports while keeping the validated modern
dependency versions.
Model downloads use the writable repository cache `.cache/medclaw/torch` by
default. Set `MEDCLAW_MODEL_CACHE_ROOT` or `TORCH_HOME` to override it.
Runtime inference does not download model weights by default. The Torch cache
must already contain `hub/checkpoints/dc_student.pth` and
`hub/checkpoints/unet_r231-d5d2fc3d.pth`; otherwise the skill fails fast instead
of hanging on offline servers. Use `scripts/prewarm_model.py` on a networked
machine, or run `scripts/diagnose_env.py` to check the cache.

## Output

The skill returns structured findings and an audit artifact set:

- Raw binary LungTumorMask segmentation NIfTI.
- Largest connected tumor component NIfTI, when a tumor is detected.
- Axial ROI PNG with a fixed 32-pixel margin.
- Full axial source-slice PNG.
- Axial overlay PNG showing the selected mask and expanded ROI box.
- JSON metadata describing the selection and model provenance.

If no tumor is predicted, the call succeeds with `tumor_detected=false` and
returns the raw mask plus metadata. This is a research benchmark tool, not a
clinical diagnostic system.
