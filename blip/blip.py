from typing import List, Union

import torch
from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from blip.model import BLIP, load_blip

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODELS = {
    "base": "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth",
}


def _convert_image_to_rgb(image: Image.Image) -> Image.Image:
    return image.convert("RGB")


def _transform(image_size: int = 224) -> Compose:
    return Compose(
        [
            Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


def available_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list(MODELS.keys())


def load(name: str = "base", device: Union[str, torch.device] = DEVICE) -> BLIP:
    """Load a BLIP model

    Parameters
    ----------
    name : str
        A model name listed by `blip.available_models()`, or the path to a model
        checkpoint containing the state_dict
    device : Union[str, torch.device]
        The device to put the loaded model

    Returns
    -------
    model : torch.nn.Module
        The CLIP model
    preprocess : Callable[[PIL.Image], torch.Tensor]
        A torchvision transform that converts a PIL image into a tensor that the
        returned model can take as its input
    """
    model = load_blip(url=MODELS[name], device=device)
    return model, _transform()
