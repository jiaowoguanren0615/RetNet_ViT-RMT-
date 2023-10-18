from functools import partial
from timm.models import register_model
from collections import OrderedDict
import torch
from retention import RMTBlock, ConvStem, Conv3X3S2
import torch.nn as nn

class RMT(nn.Module):
    def __init__(self, in_c,
                 layers, hidden_dim, heads, num_classes, representation_size=None, double_v_dim=True, **kwargs):
        super().__init__()
        self.layers = layers
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.num_classes = num_classes
        self.double_v_dim = double_v_dim
        
        self.conv_stem = ConvStem(in_channels=in_c, out_channels=self.hidden_dim[0], kernel_size=3)
        self.c3s2 = Conv3X3S2(self.hidden_dim[0], self.hidden_dim[0], 3, 1)

        self.layer1 = self._make_layer(self.layers[0], self.hidden_dim[0], block_idx=0)
        self.layer2 = self._make_layer(self.layers[1], self.hidden_dim[1], block_idx=1)
        self.layer3 = self._make_layer(self.layers[2], self.hidden_dim[2], block_idx=2)
        self.layer4 = self._make_layer(self.layers[3], self.hidden_dim[3], True, block_idx=3)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        if representation_size:
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(self.hidden_dim, self.hidden_dim)),
                ("act", nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        
        self.fc = nn.Linear(self.hidden_dim[3], self.num_classes)


    def _make_layer(self, layer_num, hidden_dim, last_block=False, block_idx=None):
        layers = []
        # in_channels, out_channels, hidden_size, heads
        self.rmt_block = RMTBlock(hidden_dim, hidden_dim, hidden_dim, self.heads)
        for idx in range(layer_num):
            layers.append((f'RMTBlock_{idx}', RMTBlock(hidden_dim, hidden_dim, hidden_dim, self.heads)))

        if last_block:
            pass
        else:
            layers.append(('Conv3S2', Conv3X3S2(hidden_dim, self.hidden_dim[block_idx+1], 3, 1)))
        return nn.Sequential(OrderedDict(layers))


    def forward_features(self, x):
        x = self.conv_stem(x)
        x = self.c3s2(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # return self.pre_logits(x[:, 0])
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x


@register_model
def RetViT_tiny(num_classes, **kwargs):
    layers = [2, 2, 2, 2]
    hidden_dim = [64, 128, 256, 384]
    heads = 16

    model = RMT(in_c=3,
                layers=layers,
                hidden_dim=hidden_dim,
                heads=heads,
                num_classes=num_classes,
                representation_size=None,
                double_v_dim=True, **kwargs)
    return model


@register_model
def RetViT_base(num_classes, **kwargs):
    layers = [3, 3, 3, 3]
    hidden_dim = [128, 256, 384, 512]
    heads = 16

    model = RMT(in_c=3,
                layers=layers,
                hidden_dim=hidden_dim,
                heads=heads,
                num_classes=num_classes,
                representation_size=None,
                double_v_dim=True, **kwargs)
    return model


@register_model
def RetViT_base(num_classes, **kwargs):
    layers = [3, 3, 3, 3]
    hidden_dim = [256, 384, 512, 640]
    heads = 16

    model = RMT(in_c=3,
                layers=layers,
                hidden_dim=hidden_dim,
                heads=heads,
                num_classes=num_classes,
                representation_size=None,
                double_v_dim=True, **kwargs)
    return model



@register_model
def RetViT_large(num_classes, **kwargs):
    layers = [4, 4, 4, 4]
    hidden_dim = [512, 640, 768, 896]
    heads = 16

    model = RMT(in_c=3,
                layers=layers,
                hidden_dim=hidden_dim,
                heads=heads,
                num_classes=num_classes,
                representation_size=None,
                double_v_dim=True, **kwargs)
    return model


@register_model
def RetViT_large(num_classes, **kwargs):
    layers = [4, 4, 4, 4]
    hidden_dim = [640, 768, 896, 1024]
    heads = 16

    model = RMT(in_c=3,
                layers=layers,
                hidden_dim=hidden_dim,
                heads=heads,
                num_classes=num_classes,
                representation_size=None,
                double_v_dim=True, **kwargs)
    return model


@register_model
def RetViT_huge(num_classes, **kwargs):
    layers = [8, 8, 8, 8]
    hidden_dim = [512, 1024, 2048, 4096]
    heads = 16

    model = RMT(in_c=3,
                layers=layers,
                hidden_dim=hidden_dim,
                heads=heads,
                num_classes=num_classes,
                representation_size=None,
                double_v_dim=True, **kwargs)
    return model



if __name__ == '__main__':
    from torchinfo import summary
    net = RetViT_tiny(num_classes=1000)
    summary(net, input_size=(1, 3, 224, 224))