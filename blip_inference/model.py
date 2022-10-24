"""
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
"""

import warnings
from typing import Dict, Tuple, Union

import torch
import torch.nn.functional as F
from timm.models.hub import download_cached_file
from torch import Tensor, nn
from transformers import BatchEncoding, BertTokenizer

from blip_inference.compat import BertConfig, BertModel
from blip_inference.vit import create_vit, interpolate_pos_embed

warnings.filterwarnings("ignore")


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BERT_CONFIG = {
    "architectures": ["BertModel"],
    "attention_probs_dropout_prob": 0.1,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "hidden_size": 768,
    "initializer_range": 0.02,
    "intermediate_size": 3072,
    "layer_norm_eps": 1e-12,
    "max_position_embeddings": 512,
    "model_type": "bert",
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "pad_token_id": 0,
    "type_vocab_size": 2,
    "vocab_size": 30524,
    "encoder_width": 768,
    "add_cross_attention": True,
}


class BLIPFeatureExtractor(nn.Module):
    def __init__(self, image_size: int = 224, vit: str = "base"):
        """
        Args:
            image_size (int): input image size
            vit (str): model size of vision transformer
        """
        super().__init__()
        self.image_size = image_size
        self.vit = vit

        visual, vision_width = create_vit(vit, image_size)
        self.visual = visual
        self.tokenizer = init_tokenizer()
        config = BertConfig.from_dict(BERT_CONFIG)
        config.encoder_width = vision_width
        self.text_encoder = BertModel(config=config, add_pooling_layer=False)

    def encode_image(self, image: Tensor) -> Tensor:
        return self.visual(image)

    def encode_text(self, text: BatchEncoding) -> Tensor:
        return self.text_encoder(
            text.input_ids,
            attention_mask=text.attention_mask,
            return_dict=True,
            mode="text",
        ).last_hidden_state

    def forward(self, image_encoding: Tensor, text_encoding: BatchEncoding) -> Tensor:
        raise NotImplementedError


def load_blip_feature_extractor(
    url: str,
    device: Union[str, torch.device] = DEVICE,
) -> BLIPFeatureExtractor:
    model = BLIPFeatureExtractor()
    model, msg = load_checkpoint(model, url, device=device)
    assert len(msg.missing_keys) == 0
    return model.eval().to(device)


def init_tokenizer() -> BertTokenizer:
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.add_special_tokens({"bos_token": "[DEC]"})
    tokenizer.add_special_tokens({"additional_special_tokens": ["[ENC]"]})
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
    return tokenizer


def load_checkpoint(
    model: nn.Module, url: str, device: Union[str, torch.device] = DEVICE
):
    cached_file = download_cached_file(url, check_hash=False, progress=True)
    checkpoint = torch.load(cached_file, map_location=device)

    state_dict: Dict[str, Tensor] = checkpoint["model"]
    # Change vision backbone keys ('vision_encoder' --> 'visual') to match CLIP API.
    for name in list(state_dict.keys()):
        if name[:14] == "visual_encoder":
            new_name = name.replace("visual_encoder", "visual")
            state_dict[new_name] = state_dict.pop(name)

    state_dict["visual.pos_embed"] = interpolate_pos_embed(
        state_dict["visual.pos_embed"], model.visual
    )

    msg = model.load_state_dict(state_dict, strict=False)
    print("load checkpoint from %s" % url)
    return model, msg


class BLIP(nn.Module):
    def __init__(
        self,
        image_size: int = 384,
        vit: str = "base",
        embed_dim: int = 256,
    ):
        super().__init__()
        self.image_size = image_size
        self.vit = vit

        visual, vision_width = create_vit(vit, image_size)
        # TODO: Create wrapper (nn.Module) to get features, extract one feature token,
        # and combine with 'self.vision_proj' below.
        self.visual = visual
        self.tokenizer = init_tokenizer()
        med_config = BertConfig.from_dict(BERT_CONFIG)
        med_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False)

        text_width = self.text_encoder.config.hidden_size

        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

    def encode_image(self, image: Tensor) -> Tensor:
        x = self.visual(image)[:, 0, :]
        return F.normalize(self.vision_proj(x), dim=-1)

    def encode_text(self, text: BatchEncoding) -> Tensor:
        x = self.text_encoder(
            text.input_ids,
            attention_mask=text.attention_mask,
            return_dict=True,
            mode="text",
        ).last_hidden_state[:, 0, :]
        return F.normalize(self.text_proj(x), dim=-1)

    def forward(self, image: Tensor, text: BatchEncoding) -> Tuple[Tensor, Tensor]:
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        logits_per_image = 100.0 * image_features @ text_features.T
        logits_per_text = logits_per_image.T
        return logits_per_image, logits_per_text


def load_blip(
    url: str,
    device: Union[str, torch.device] = DEVICE,
    vit: str = "base",
):
    model = BLIP(vit=vit)
    model, msg = load_checkpoint(model, url, device=device)
    assert len(msg.missing_keys) == 0
    return model.eval().to(device)
