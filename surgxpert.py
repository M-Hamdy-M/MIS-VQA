import gzip
import json
import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import resnet50
from transformers import CLIPVisionModel, SamModel, SegformerConfig, SegformerModel

import llava.model.llava_arch as llava_arch

WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")


class Processor:
    image_mean = [0, 0, 0]

    def preprocess(self, images, return_tensors="pt"):
        images = images if isinstance(images, list) else [images]
        return {"pixel_values": torch.stack([T.ToTensor()(T.CenterCrop(1024)(i)) for i in images])}

    __call__ = preprocess


class Tower(nn.Module):
    def __init__(self, name, delay_load=False):
        super().__init__()
        self.name, self.is_loaded, self.image_processor = name, False, Processor()
        if not delay_load:
            self.load_model()

    def load_model(self, device_map=None):
        if self.is_loaded:
            return
        self.build()
        self.requires_grad_(False).eval()
        if device_map == "auto" and torch.cuda.is_available():
            self.cuda()
        self.is_loaded = True

    def forward(self, x):
        y = x if self.size == 1024 else F.interpolate(x.float(), self.size, mode="bilinear", align_corners=True).to(x.dtype)
        y = self.features(y.to(next(self.parameters()).dtype)).to(x.dtype)
        if y.shape[-1] != 32:
            y = F.interpolate(y.float(), 32, mode="bilinear", align_corners=True).to(x.dtype)
        return y.flatten(2).transpose(1, 2).contiguous()


class CLIP(Tower):
    size, hidden_size = 336, 768

    def build(self):
        self.model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16").vision_model

    def features(self, x):
        m = self.model
        x = m.embeddings.patch_embedding(x).flatten(2).transpose(1, 2)
        pos = m.embeddings.position_embedding.weight[1:]
        n, s = int(len(pos) ** 0.5), int(x.shape[1] ** 0.5)
        pos = pos.float().reshape(1, n, n, -1).permute(0, 3, 1, 2)
        pos = F.interpolate(pos, s, mode="bicubic", antialias=True).permute(0, 2, 3, 1).reshape(1, s * s, -1)
        x = m.pre_layrnorm(x + pos.to(x.dtype))
        for layer in m.encoder.layers:
            x = layer(x, None, None)[0]
        return x.transpose(1, 2).reshape(len(x), -1, s, s)


class SAM(Tower):
    size, hidden_size = 1024, 1024

    def build(self):
        self.model = SamModel.from_pretrained("facebook/sam-vit-base").vision_encoder
        self.neck, self.model.neck = self.model.neck, nn.Identity()

    def features(self, x):
        x = self.model(x).last_hidden_state.permute(0, 3, 1, 2)
        return F.pixel_unshuffle(self.neck.layer_norm1(self.neck.conv1(x)), 2)


class LLFDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        for name, i, o, k in [("proj1", 64, 128, 1), ("proj2", 128, 128, 1), ("proj3", 320, 128, 3), ("proj4_1", 512, 256, 3),
                              ("proj4_2", 256, 128, 3), ("projSkips_1", 512, 256, 1), ("projSkips_2", 256, 128, 1)]:
            setattr(self, name, nn.Conv2d(i, o, k, padding=k // 2))
            setattr(self, "bn_" + name, nn.BatchNorm2d(o))
        self.attention1 = nn.MultiheadAttention(128, 2, batch_first=True)
        self.attention2 = nn.MultiheadAttention(128, 4, batch_first=True)

    def block(self, name, x):
        return F.leaky_relu(getattr(self, "bn_" + name)(getattr(self, name)(x)))

    def attend(self, attention, x):
        q = x.flatten(2).transpose(1, 2)
        return attention(q, q, q)[0].transpose(1, 2).contiguous().view(x.shape)

    def up(self, x, *sizes):
        for s in sizes:
            x = F.interpolate(x, s, mode="bilinear", align_corners=False)
        return x

    def forward(self, hs):
        x1 = self.attend(self.attention1, self.block("proj1", hs[0]))
        x2 = self.up(self.attend(self.attention2, self.block("proj2", hs[1])), 128)
        x3 = self.up(self.block("proj3", hs[2]), 64, 128)
        x4 = self.up(self.block("proj4_2", self.block("proj4_1", hs[3])), 32, 64, 128)
        return self.block("projSkips_2", self.block("projSkips_1", torch.cat([x1, x2, x3, x4], 1)))


class FASL(Tower):
    size, hidden_size = 512, 1024

    def build(self):
        self.segformer = SegformerModel(SegformerConfig.from_pretrained("nvidia/mit-b5", output_hidden_states=True))
        self.llf_decoder, self.conv1LLF, self.bn_conv1LLF = LLFDecoder(), nn.Conv2d(128, 64, 1), nn.BatchNorm2d(64)
        weights = torch.load(f"{WEIGHTS}/{self.name.replace('-', '_')}.pth", map_location="cpu")
        weights = weights.get("model_state_dict", weights)
        self.load_state_dict({k: v for k, v in weights.items() if k.split(".")[0] in dict(self.named_children())})

    def features(self, x):
        x = self.llf_decoder(self.segformer(x).hidden_states)
        return F.pixel_unshuffle(F.leaky_relu(self.bn_conv1LLF(self.conv1LLF(x))), 4)


class PeskaVLP(Tower):
    size, hidden_size = 224, 128

    def build(self):
        self.model = nn.Sequential(OrderedDict(list(resnet50().named_children())[:-2]))
        weights = torch.load(f"{WEIGHTS}/PeskaVLP.pth", map_location="cpu")
        self.model.load_state_dict({k[19:]: v for k, v in weights.items() if k.startswith("backbone_img.model.")})

    def features(self, x):
        return F.pixel_shuffle(self.model(x), 4)


TOWERS = {"clip-336-b16": CLIP, "sam-b": SAM, "fasl-cholec": FASL, "fasl-endovis18": FASL, "peskavlp": PeskaVLP}


def build_vision_tower(cfg, **kwargs):
    name = getattr(cfg, "mm_vision_tower", getattr(cfg, "vision_tower", None)).strip(";")
    return TOWERS[name](name, **kwargs)


llava_arch.build_vision_tower = build_vision_tower

PROMPTS = {"concise": "Describe the surgical scene concisely.", "detailed": "Describe the surgical scene in detail."}


def to_llava(src, dst, types=("concise", "detailed", "conv1", "conv2", "reasoning")):
    out = []
    for s in json.load(gzip.open(src)):
        for t in types:
            for q, a in [(PROMPTS[t], s[t])] if t in PROMPTS else s[t]:
                out.append({"id": f"{t}_{len(out)}", "image": s["image"], "conversations": [
                    {"from": "human", "value": "<image>\n" + q}, {"from": "gpt", "value": a}]})
    json.dump(out, open(dst, "w"))


if __name__ == "__main__":
    from llava.train.train import train
    train(attn_implementation="eager")
