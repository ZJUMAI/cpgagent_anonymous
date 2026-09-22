# pathology.ucec_conch_patch_roi

Select UCEC pathology patch ROIs using cached CONCH v1.5 prompt-ranking outputs.
The normal agent call only needs `case_id`.

## What It Does

1. Locate the case directory from `MEDCLAW_CASES_ROOT` or
   `examples/cases/{case_id}`.
2. Resolve `pathology.conch_roi_dir` from `case.yaml`, then search the standard
   `pathology/roi_256`, `pathology/roi_512`, and `pathology/conch_roi` roots for
   a slide-specific CONCH ROI packet containing `prompt_scores.json`.
3. Select the highest-scoring patch images for UCEC-oriented prompts
   (`endometrioid_tumor`, `tumor_glands` by default).
4. Export individual patch ROI PNGs, a labeled contact sheet, and
   `ucec_conch_patch_roi_metadata.json` for audit.

Offline preprocessing mirrors LUAD: CONCH patch features, prompt scoring, and
WSI ROI crop are written under `processed/UCEC/{case_id}/pathology/conch_roi/`.

The tool never sends the WSI or CONCH feature files to the external model. Only
the exported PNG patch ROI artifacts are eligible for multimodal feedback.
If cached prompt IDs differ from the requested IDs, the tool records a warning
and falls back to the packet's first ranked prompts. Oversized ancillary PNG
text metadata is discarded without changing Pillow's global safety limits.

## Inputs

- `case_id` (required): benchmark case id.
- `slide_stem` (optional): use a specific slide when multiple slides exist.
- `prompt_ids` (optional): list of prompt ids to inspect.
- `top_k_per_prompt` (optional): 1-10 patch ROIs per prompt, default 3.
- `conch_roi_uri` (optional): trusted override to a case or slide CONCH ROI
  directory.

## Outputs

- `patch_contact_sheet.png` image artifact.
- One `patch_roi` image artifact per selected patch.
- `ucec_conch_patch_roi_metadata.json` with slide stem, prompt ids, top scores,
  coordinates, patch indices, image filenames, and source packet paths.

This is research benchmark output only and must not be presented as clinical
advice.
