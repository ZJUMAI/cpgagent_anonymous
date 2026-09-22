"""Download and validate the LungTumorMask and lungmask model caches."""

from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lungmask import LMInferer

from medclaw.skills.radiology.lung_tumor_roi.compat import (  # noqa: E402
    configure_model_cache,
    prepare_lungtumormask_import,
)


def main() -> int:
    os.environ.setdefault("MEDCLAW_LUNG_TUMOR_ALLOW_MODEL_DOWNLOAD", "1")
    cache = configure_model_cache()
    prepare_lungtumormask_import()

    from lungtumormask.mask import load_model

    print(f"Model cache: {cache}")
    model = load_model()
    print(f"LungTumorMask loaded: {model.__class__.__name__}")
    inferer = LMInferer()
    print(f"lungmask loaded: {inferer.__class__.__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
