import torch
import torch.nn as nn
import retnet
from torchinfo import summary

if __name__ == "__main__":
    # verify model size for hyperparameters in paper
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1.3B model
    layers = 24
    hidden_dim = 768
    ffn_size = 4096
    heads = 16

    retnet1 = retnet.RetNet(layers, hidden_dim, ffn_size, heads, double_v_dim=True)
    print("1.3B model:",sum(p.numel() for p in retnet1.parameters() if p.requires_grad))

    summary(retnet1, input_size=(2, 576, 768))