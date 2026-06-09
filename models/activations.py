import torch
import torch.nn as nn

def get_activations(name:str="relu"):
    name = name.lower()
    if name == "relu":
        return nn.ReLU
    if name == "celu":
        return nn.CELU
    if name == "leakyrelu":
        return nn.LeakyReLU
    if name == "sigmoid":
        return nn.Sigmoid
    if name == 'silu':
        return nn.SiLU
    if name == 'x':
        return nn.Identity
    if name == 'tanh':
        return nn.Tanh
    if name == 'prelu':
        return nn.PReLU
    if name == "elu":
        return nn.ELU
    
    raise ValueError("")
