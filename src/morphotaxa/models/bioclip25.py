from __future__ import annotations

import os


def load_bioclip25(model_name: str = "hf-hub:imageomics/bioclip-2.5-vith14", cache_dir: str | None = None, offline: bool = True):
    """Load BioCLIP 2.5 through open_clip.

    On the user's AutoDL setup, the Hugging Face cache should already contain
    the approved/public model files. This function does not implement any
    authorization bypass and can run in offline-cache mode.
    """
    if offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    try:
        import open_clip
    except ImportError as e:
        raise RuntimeError("open_clip is required; activate the medplant26 environment") from e
    kwargs = {}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(model_name, **kwargs)
    return model, preprocess_train, preprocess_val
