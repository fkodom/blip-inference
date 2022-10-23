"""
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
"""

import warnings
from typing import Dict, Union

import torch
from timm.models.hub import download_cached_file
from torch import Tensor, nn
from transformers import BertTokenizer

from blip.compat import BertConfig, BertModel
from blip.vit import create_vit, interpolate_pos_embed

warnings.filterwarnings("ignore")


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRETRAINED_WEIGHTS_URL = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth"
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


class BLIP(nn.Module):
    def __init__(self, image_size: int = 224, vit: str = "base"):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """
        super().__init__()
        visual, vision_width = create_vit(vit, image_size)
        self.visual = visual
        self.tokenizer = init_tokenizer()
        config = BertConfig.from_dict(BERT_CONFIG)
        config.encoder_width = vision_width
        self.text_encoder = BertModel(config=config, add_pooling_layer=False)

    def forward(self, image, caption, mode):
        assert mode in [
            "image",
            "text",
            "multimodal",
        ], "mode parameter must be image, text, or multimodal"
        text = self.tokenizer(caption, return_tensors="pt").to(image.device)

        # TODO: Split these out into separate 'encode_image', 'encode_text' methods
        if mode == "image":
            # return image features
            image_embeds = self.visual(image)
            return image_embeds

        elif mode == "text":
            # return text features
            text_output = self.text_encoder(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
                mode="text",
            )
            return text_output.last_hidden_state

        elif mode == "multimodal":
            # return multimodel features
            image_embeds = self.visual(image)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                image.device
            )

            text.input_ids[:, 0] = self.tokenizer.enc_token_id
            output = self.text_encoder(
                text.input_ids,
                attention_mask=text.attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            return output.last_hidden_state


def load_blip(
    url: str = PRETRAINED_WEIGHTS_URL,
    device: Union[str, torch.device] = DEVICE,
) -> BLIP:
    model = BLIP()
    model, msg = load_checkpoint(model, url, device=device)
    assert len(msg.missing_keys) == 0
    return model


def init_tokenizer() -> BertTokenizer:
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.add_special_tokens({"bos_token": "[DEC]"})
    tokenizer.add_special_tokens({"additional_special_tokens": ["[ENC]"]})
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
    return tokenizer


def load_checkpoint(model: BLIP, url: str, device: Union[str, torch.device] = DEVICE):
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
