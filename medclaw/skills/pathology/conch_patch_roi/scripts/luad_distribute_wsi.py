#!/usr/bin/env python3
"""Place LUAD WSI slides under processed/LUNG/{case_id}/pathology/wsi/."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from _luad_multiscale_common import case_id_from_stem, stem_from_name


def link_or_copy(src: Path, dst: Path, method: str) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if method == "copy":
            if dst.is_file() and not dst.is_symlink():
                if dst.stat().st_size == src.stat().st_size:
                    return "skip"
            else:
                dst.unlink()
        elif dst.is_symlink() or dst.is_file():
            try:
                if dst.resolve() == src.resolve():
                    return "skip"
            except OSError:
                pass
            if dst.is_file() and not dst.is_symlink() and dst.stat().st_size == src.stat().st_size:
                return "skip"
            dst.unlink()

    methods = [method] if method != "auto" else ("hardlink", "symlink", "copy")
    last_err: OSError | None = None
    for m in methods:
        try:
            if m == "hardlink":
                os.link(src, dst)
            elif m == "symlink":
                dst.symlink_to(src)
            elif m == "copy":
                import shutil

                shutil.copy2(src, dst)
            else:
                raise ValueError(f"unknown method: {m}")
            return m
        except OSError as e:
            last_err = e
            if dst.exists() or dst.is_symlink():
                dst.unlink()
    assert last_err is not None
    raise last_err


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--wsi-dir",
        type=Path,
        default=Path("/data4/share/TCGA/TCGA_LUAD/wsi"),
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        default=Path("/data4/tujiayong/processed/LUNG"),
    )
    ap.add_argument(
        "--method",
        choices=("auto", "hardlink", "symlink", "copy"),
        default="auto",
        help="auto tries hardlink then symlink then copy",
    )
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if not args.wsi_dir.is_dir():
        ap.error(f"wsi dir not found: {args.wsi_dir}")

    slides = sorted(args.wsi_dir.glob("*.svs"))
    if args.limit > 0:
        slides = slides[: args.limit]

    counts: dict[str, int] = {}
    for src in slides:
        stem = stem_from_name(src.name)
        case_id = case_id_from_stem(stem)
        dst = args.out_root / case_id / "pathology" / "wsi" / src.name
        try:
            status = link_or_copy(src, dst, args.method)
        except OSError as e:
            print(f"FAIL {src.name} -> {dst}: {e}")
            counts["fail"] = counts.get("fail", 0) + 1
            continue
        counts[status] = counts.get(status, 0) + 1
        if status != "skip":
            print(f"{status:8} {case_id}/pathology/wsi/{src.name}")

    print(
        f"\nDone: {len(slides)} slides | "
        + " ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    )
    return 0 if counts.get("fail", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
