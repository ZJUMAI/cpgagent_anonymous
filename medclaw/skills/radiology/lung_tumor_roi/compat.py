"""Compatibility helpers for importing LungTumorMask with modern MONAI."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def configure_model_cache() -> Path:
    """Configure a writable Torch cache used by LungTumorMask and lungmask."""

    project_root = Path(__file__).resolve().parents[4]
    cache_root = Path(
        os.environ.get("MEDCLAW_MODEL_CACHE_ROOT")
        or project_root / ".cache" / "medclaw"
    ).expanduser()
    torch_home = Path(os.environ.get("TORCH_HOME") or cache_root / "torch").expanduser()
    torch_home.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(torch_home.resolve())
    return torch_home.resolve()


def prepare_lungtumormask_import() -> None:
    """Provide MONAI transforms that LungTumorMask 1.3.1 imports by old names."""

    import monai.transforms as transforms
    import monai.utils as utils

    if not hasattr(transforms, "AddChanneld"):
        from monai.transforms import EnsureChannelFirstd

        class AddChanneld(EnsureChannelFirstd):
            """Modern equivalent of the removed dictionary AddChannel transform."""

            def __init__(
                self,
                keys: Any,
                allow_missing_keys: bool = False,
                **_: Any,
            ) -> None:
                super().__init__(
                    keys=keys,
                    channel_dim="no_channel",
                    allow_missing_keys=allow_missing_keys,
                )

        transforms.AddChanneld = AddChanneld

    for name in ("alias", "export"):
        if not hasattr(utils, name):
            setattr(utils, name, _passthrough_decorator)


def _passthrough_decorator(*args: Any, **kwargs: Any) -> Any:
    """Return an object unchanged for removed MONAI metadata decorators."""

    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]

    def decorate(value: Any) -> Any:
        return value

    return decorate
