"""Diagnose the lung tumor ROI runtime without running full CT inference."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from medclaw.skills.radiology.lung_tumor_roi.compat import (  # noqa: E402
    configure_model_cache,
    prepare_lungtumormask_import,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--load-models",
        action="store_true",
        help="Actually instantiate LungTumorMask and lungmask models.",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow torch.hub downloads while loading models.",
    )
    args = parser.parse_args()

    if args.allow_download:
        os.environ["MEDCLAW_LUNG_TUMOR_ALLOW_MODEL_DOWNLOAD"] = "1"

    cache = configure_model_cache()
    print(f"TORCH_HOME={cache}")

    import torch

    print(f"torch={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda_device_count={torch.cuda.device_count()}")
        print(f"cuda_device_0={torch.cuda.get_device_name(0)}")

    prepare_lungtumormask_import()
    import lungtumormask  # noqa: F401
    import lungmask  # noqa: F401

    print("imports=ok")

    from medclaw.skills.radiology.lung_tumor_roi.inference import (
        require_cached_model_weights,
        required_model_weight_paths,
    )

    paths = required_model_weight_paths(cache)
    for label, path in paths.items():
        print(f"weight[{label}]={path} exists={path.is_file()}")

    try:
        require_cached_model_weights(cache)
    except RuntimeError as exc:
        print(f"weights=missing: {exc}")
        return 2
    print("weights=ok")

    if args.load_models:
        from lungmask import LMInferer
        from lungtumormask.mask import load_model

        print("loading_lungtumormask=begin", flush=True)
        load_model()
        print("loading_lungtumormask=ok", flush=True)
        print("loading_lungmask_R231=begin", flush=True)
        LMInferer(modelname="R231", force_cpu=not torch.cuda.is_available())
        print("loading_lungmask_R231=ok", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
