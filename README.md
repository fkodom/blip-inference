# blip-inference

Pretrained [BLIP](https://github.com/salesforce/BLIP) with a similar API to [CLIP](https://github.com/openai/CLIP).

BLIP tends to achieve slightly better accuracy than CLIP with similar inference speed.  The CLIP API is much cleaner and more commonly used.  This repo refactors BLIP to match the CLIP interface, so that it's easier for practitioners to switch between CLIP / BLIP models.


## Install

From PyPI:
```bash
pip install blip-inference
```

From source:
```bash
pip install "blip_inference @ git+https://git@github.com/fkodom/blip-inference.git"
```


## Usage

User-facing methods behave similarly to CLIP.  A few underlying details change, which will only affect advanced users.

```python
import torch
import blip_inference as blip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = blip.load("feature_extractor", device=device)

raw_text = ["a diagram", "a dog", "a cat"]
text = blip.tokenize(raw_text).to(device)
image = preprocess(Image.open("kitten.jpeg")).unsqueeze(0).to(device)

with torch.no_grad():    
    logits_per_image, logits_per_text = model(image, text)
probs = torch.softmax(logits_per_image, dim=-1)

print("\nPredictions:\n")
for idx, value in enumerate(probs.squeeze()):
    print(f"{raw_text[idx]:>16s}: {100 * value.item():.2f}%")
probs = logits_per_image.softmax(dim=-1).cpu().numpy()
```

### Zero-Shot Prediction

```python
import blip_inference as blip
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = blip.load('base', device)

raw_text = ["a diagram", "a dog", "a cat"]
text = blip.tokenize(raw_text).to(device)
image = preprocess(Image.open("kitten.jpeg")).unsqueeze(0).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = (100 * image_features @ text_features.T).softmax(dim=-1)

print("\nPredictions:\n")
for idx, value in enumerate(similarity.squeeze()):
    print(f"{raw_text[idx]:>16s}: {100 * value.item():.2f}%")
```

### Linear Probe Evaluation

See [this example from the CLIP repo](https://github.com/openai/CLIP#linear-probe-evaluation).  Everything should be identical, except for swapping:
* `import clip` --> `import blip_inference as blip`
* `clip` --> `blip`


## API

Similar to CLIP, the `blip_inference` module provides the following methods:

#### `blip_inference.available_models() -> List[str]`

Returns the names of the available BLIP models.

#### `blip_inference.load(name: str, device=...) -> Tuple[BLIP, Callable]`

Returns the model and the TorchVision transform needed by the model, specified by the model name returned by `blip_inference.available_models()`. It will download the model as necessary. The `name` argument can also be a path to a local checkpoint.

The device to run the model can be optionally specified, and the default is to use the first CUDA device if there is any, otherwise the CPU.

#### `blip_inference.tokenize(text: Union[str, List[str]], context_length: int = 35) -> BatchEncoding`

Returns a dictionary with tokenized sequences of given text input(s). This can be used as the input to the model

---

The model returned by `blip_inference.load()` supports the following methods:

#### `model.encode_image(image: Tensor) -> Tensor`

Given a batch of images, returns the image features encoded by the vision portion of the BLIP model.

#### `model.encode_text(text: BatchEncoding) -> Tensor`

Given a batch of text tokens, returns the text features encoded by the language portion of the BLIP model.

#### `model(image: Tensor, text: BatchEncoding) -> Tuple[Tensor, Tensor]`

Given a batch of images and a batch of text tokens, returns two Tensors, containing the logit scores corresponding to each image and text input. The values are cosine similarities between the corresponding image and text features.

**NOTE**: Unlike CLIP, logits for BLIP models **do not** need to be multiplied by 100 before computing cosine similarity.  That scaling factor is built into the BLIP model. 