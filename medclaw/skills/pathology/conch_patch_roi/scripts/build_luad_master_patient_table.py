#!/usr/bin/env python3
"""Build LUAD master patient table from RAW share paths only (not processed/)."""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

PATIENT_RE = re.compile(r"^(TCGA-[A-Z0-9]+-[A-Z0-9]+)")


def patient_id(name: str) -> str | None:
    m = PATIENT_RE.match(name)
    return m.group(1) if m else None


def collect_tcia_dirs(root: Path) -> set[str]:
    out: set[str] = set()
    if not root.exists():
        return out
    for p in root.iterdir():
        if p.is_dir() and not p.name.startswith("._"):
            pid = patient_id(p.name)
            if pid:
                out.add(pid)
    return out


def assign_cohort(has_radiology: bool, has_pathology: bool) -> tuple[str, str]:
    """Return (cohort_group, cohort_label) for metadata table."""
    if has_radiology and has_pathology:
        return "both", "双模态_CT+病理"
    if has_pathology:
        return "pathology_only", "仅病理"
    if has_radiology:
        return "radiology_only", "仅影像"
    return "neither", "无本地数据"


def collect_slide_patients(
    root: Path, pattern: str
) -> tuple[set[str], dict[str, list[str]]]:
    patients: set[str] = set()
    slides: dict[str, list[str]] = defaultdict(list)
    if not root.exists():
        return patients, slides
    for f in root.glob(pattern):
        pid = patient_id(f.stem)
        if pid:
            patients.add(pid)
            slides[pid].append(f.name)
    return patients, slides


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Master table from /data4/share raw LUAD paths"
    )
    ap.add_argument(
        "--tcia-root",
        type=Path,
        default=Path("/data4/share/TCIA/tcga_luad"),
    )
    ap.add_argument(
        "--wsi-dir",
        type=Path,
        default=Path("/data4/share/TCGA/TCGA_LUAD/wsi"),
    )
    ap.add_argument(
        "--patch-dir",
        type=Path,
        default=Path("/data4/share/TCGA/TCGA_LUAD/patches/patches"),
    )
    ap.add_argument(
        "--conch-dir",
        type=Path,
        default=Path("/data4/share/CONCH_TCGA/LUAD/features/pt_files"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("/data4/tujiayong/processed/LUNG/luad_master_patient_table.tsv"),
    )
    args = ap.parse_args()

    tcia_pts = collect_tcia_dirs(args.tcia_root)
    wsi_pts, wsi_slides = collect_slide_patients(args.wsi_dir, "*.svs")
    patch_pts, patch_slides = collect_slide_patients(args.patch_dir, "*.h5")
    conch_pts, _ = collect_slide_patients(args.conch_dir, "*.pt")

    all_ids = sorted(tcia_pts | wsi_pts | patch_pts | conch_pts)

    rows = []
    for cid in all_ids:
        has_tcia = cid in tcia_pts
        has_wsi = cid in wsi_pts
        has_patch = cid in patch_pts
        has_conch = cid in conch_pts
        has_radiology = has_tcia
        # 病理：盘上 WSI/patch；仅有 CONCH 特征也算病理侧（无 TCIA CT）
        has_pathology = has_wsi or has_patch or has_conch
        cohort_group, cohort_label = assign_cohort(has_radiology, has_pathology)
        rows.append(
            {
                "case_id": cid,
                "cohort_group": cohort_group,
                "cohort_label": cohort_label,
                "tcga_center": cid.split("-")[1] if cid.startswith("TCGA-") else "",
                "has_radiology": int(has_radiology),
                "has_pathology": int(has_pathology),
                "has_tcia_ct": int(has_tcia),
                "tcia_ct_path": str(args.tcia_root / cid) if has_tcia else "",
                "has_tcga_wsi": int(has_wsi),
                "n_wsi_slides": len(wsi_slides.get(cid, [])),
                "wsi_dir": str(args.wsi_dir),
                "has_tcga_patches": int(has_patch),
                "n_patch_h5": len(patch_slides.get(cid, [])),
                "patch_dir": str(args.patch_dir),
                "has_conch_features": int(has_conch),
                "conch_feat_dir": str(args.conch_dir),
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        w.writerows(rows)

    counts = Counter(r["cohort_group"] for r in rows)
    print(f"Master table (raw share paths) -> {args.out}")
    print(f"  TCIA CT:    {len(tcia_pts)}  {args.tcia_root}")
    print(f"  TCGA WSI:   {len(wsi_pts)}  {args.wsi_dir}")
    print(f"  patches:    {len(patch_pts)}  {args.patch_dir}")
    print(f"  CONCH feat: {len(conch_pts)}  {args.conch_dir}")
    print("  cohort_group:")
    for k in ("both", "pathology_only", "radiology_only", "neither"):
        if counts[k]:
            print(f"    {k}: {counts[k]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
