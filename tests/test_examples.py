import pytest
import torch
from PIL import Image

import blip

device = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(params=["base", "large"])
@pytest.mark.parametrize("name", ["base", "large"])
def test_usage(name: str):
    model, preprocess = blip.load(name, device=device)

    raw_text = ["a diagram", "a dog", "a cat"]
    text = blip.tokenize(raw_text).to(device)
    image = preprocess(Image.open("kitten.jpeg")).unsqueeze(0).to(device)

    with torch.no_grad():
        logits_per_image, _ = model(image, text)
    probs = torch.softmax(logits_per_image, dim=-1)

    print("\nPredictions:\n")
    for idx, value in enumerate(probs.squeeze()):
        print(f"{raw_text[idx]:>16s}: {100 * value.item():.2f}%")
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()


@pytest.mark.parametrize("name", ["base", "large"])
def test_zero_shot(name: str):
    model, preprocess = blip.load(name, device)

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


def test_available_models():
    available = blip.available_models()
    assert isinstance(available, list)
    assert len(available) > 0
