import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_libero_example() -> dict:
    """Creates a random input example for the policy."""
    return {
        "observation/state": np.random.rand(7),  # your state dim = 7
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class ActiveFlyInputs(transforms.DataTransformFn):
    """
    Convert dataset/env inputs into the model-expected format.
    Used for both training and inference.
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Your dataset has only one image stream.
        base_image = _parse_image(data["observation/image"])

        # For pi0 we can mask padded wrist images out; for pi0-fast we keep mask=True.
        pad_wrist_mask = np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": np.zeros_like(base_image),
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": pad_wrist_mask,
                "right_wrist_0_rgb": pad_wrist_mask,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class ActiveFlyOutputs(transforms.DataTransformFn):
    """
    Convert model outputs back to env/action format (inference path).
    """

    def __call__(self, data: dict) -> dict:
        # your action dim = 5
        return {"actions": np.asarray(data["actions"][:, :5])}


LiberoInputs = ActiveFlyInputs
LiberoOutputs = ActiveFlyOutputs
