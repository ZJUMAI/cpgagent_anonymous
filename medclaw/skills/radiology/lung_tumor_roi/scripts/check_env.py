"""Validate imports required by the lung tumor ROI skill."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from medclaw.skills.radiology.lung_tumor_roi.compat import (  # noqa: E402
    prepare_lungtumormask_import,
)


def main() -> int:
    prepare_lungtumormask_import()

    import lungtumormask  # noqa: F401
    import torch

    print(f"torch import ok: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available. Install a CUDA-enabled Torch build or set "
            "MEDCLAW_LUNG_TUMOR_DEVICE=cpu for an explicit CPU fallback."
        )
    print("lungtumormask import ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
