from functools import partial
from collections import OrderedDict
import torch
from retnet import RetNet
import torch.nn as nn
from torchinfo import summary


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class RMT(nn.Module):
    """
    0.26B model parameters
    layers = 24
    hidden_dim = 768
    ffn_size = 4096
    heads = 16
    representation_size=None
    double_v_dim=True
    """

    def __init__(self, layers, hidden_dim, ffn_size, heads, num_classes, representation_size=None, double_v_dim=True,
                 **kwargs):
        super().__init__()
        self.layers = layers
        self.hidden_dim = hidden_dim
        self.ffn_size = ffn_size
        self.heads = heads
        self.num_classes = num_classes
        self.double_v_dim = double_v_dim

        self.patch_embed = PatchEmbed()
        self.retnet = RetNet(self.layers, self.hidden_dim, self.ffn_size, self.heads, self.double_v_dim)

        if representation_size:
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(self.hidden_dim, self.hidden_dim)),
                ("act", nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        self.fc = nn.Linear(self.hidden_dim, self.num_classes)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.retnet(x)
        return self.pre_logits(x[:, 0])

    def forward(self, x):
        x = self.forward_features(x)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 0.26B model
    layers = 24
    hidden_dim = 768
    ffn_size = 4096
    heads = 16
    num_classes = 1000

    rmt = RMT(layers, hidden_dim, ffn_size, heads, num_classes, double_v_dim=True)
    print("0.26B model:", sum(p.numel() for p in rmt.parameters() if p.requires_grad))
    summary(rmt, input_size=(1, 3, 224, 224))