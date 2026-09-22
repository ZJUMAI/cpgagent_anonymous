# pathology.conch_patch_roi

Select pathology patch ROIs for a lung cancer case using cached CONCH v1.5
prompt-ranking outputs. The normal agent call only needs `case_id`.

## What It Does

1. Locate the case directory from `MEDCLAW_CASES_ROOT` or
   `examples/cases/{case_id}`.
2. Locate the slide-specific CONCH ROI packet containing `prompt_scores.json`.
   The default search order is `pathology/roi_256/`, `pathology/roi_512/`,
   then the legacy `pathology/conch_roi/`. A `case.yaml`
   `pathology.conch_roi_dir` or `conch_roi_uri` can still override this.
3. Select the highest-scoring patch images for tumor-oriented prompts
   (`tumor`, `tumor_nests` by default).
4. Export a low-resolution WSI overview PNG when the source slide is available,
   individual patch ROI PNGs, a labeled contact sheet, and
   `conch_patch_roi_metadata.json` for audit.

The tool never sends the WSI or CONCH feature files to the external model. Only
the exported PNG patch ROI artifacts are eligible for multimodal feedback.

## Inputs

- `case_id` (required): benchmark case id.
- `slide_stem` (optional): use a specific slide when multiple slides exist.
- `prompt_ids` (optional): list of prompt ids to inspect.
- `top_k_per_prompt` (optional): 1-10 patch ROIs per prompt, default 3.
- `conch_roi_uri` (optional): trusted override to a case or slide CONCH ROI
  directory.

## Outputs

- `wsi_overview.png` image artifact when a source WSI can be opened.
- `patch_contact_sheet.png` image artifact.
- One `patch_roi` image artifact per selected patch.
- `conch_patch_roi_metadata.json` with slide stem, prompt ids, top scores,
  coordinates, patch indices, image filenames, and source packet paths.

This is research benchmark output only and must not be presented as clinical
advice.
