import pytest
import torch
from PIL import Image

import blip_inference

device = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.mark.parametrize("name", ["base", "large"])
def test_usage(name: str):
    model, preprocess = blip_inference.load(name, device=device)

    raw_text = ["a diagram", "a dog", "a cat"]
    text = blip_inference.tokenize(raw_text).to(device)
    image = preprocess(Image.open("kitten.jpeg")).unsqueeze(0).to(device)

    with torch.no_grad():
        logits_per_image, _ = model(image, text)
    _ = torch.softmax(logits_per_image, dim=-1)


@pytest.mark.parametrize("name", ["base", "large"])
def test_zero_shot(name: str):
    model, preprocess = blip_inference.load(name, device)

    raw_text = ["a diagram", "a dog", "a cat"]
    text = blip_inference.tokenize(raw_text).to(device)
    image = preprocess(Image.open("kitten.jpeg")).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    _ = (100 * image_features @ text_features.T).softmax(dim=-1)

    _, _ = model(image, text)


def test_feature_extractor():
    model, preprocess = blip_inference.load("feature_extractor", device)

    raw_text = ["a diagram", "a dog", "a cat"]
    text = blip_inference.tokenize(raw_text).to(device)
    image = preprocess(Image.open("kitten.jpeg")).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)


def test_available_models():
    available = blip_inference.available_models()
    assert isinstance(available, list)
    assert len(available) > 0
