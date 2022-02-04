from contextlib import contextmanager
import numpy as np
import math
import h5py
import torch
from torch import nn as nn

from .training import eval_state


def get_module_output(model, input_shape, use_cuda=True):
    """
    Return the output shape of the model when fed in an array of `input_shape`.
    Note that a zero array of shape `input_shape` is fed into the model and the
    shape of the output of the model is returned.

    Args:
        model (nn.Module): PyTorch module for which to compute the output shape
        input_shape (tuple): Shape specification for the input array into the model
        use_cuda (bool, optional): If True, model will be evaluated on CUDA if available. Othewrise
            model evaluation will take place on CPU. Defaults to True.

    Returns:
        tuple: output shape of the model

    """
    # infer the original device
    initial_device = next(iter(model.parameters())).device
    device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
    with eval_state(model):
        with torch.no_grad():
            input = torch.zeros(1, *input_shape[1:], device=device)
            output = model.to(device)(input)
    model.to(initial_device)
    return output.shape


@contextmanager
def no_transforms(dat):
    """
    Contextmanager for the dataset object. It temporarily removes the transforms.
    Args:
        dat: Dataset object. Either FileTreeDataset or StaticImageSet

    Yields: The dataset object without transforms
    """
    transforms = dat.transforms
    try:
        dat.transforms = []
        yield dat
    finally:
        dat.transforms = transforms


def anscombe(x):
    """Compute Anscombe transform."""
    return 2 * np.sqrt(x + 3 / 8)
class PositionalEncoding2D(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000, learned=False, width=None, height=None):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        if width is None:
            width = height = max_len

        if learned:
            self.twod_pe = nn.Parameter(torch.randn(d_model, (height * width)))
        else:
            d_model = d_model // 2
            pe = torch.zeros(width, d_model)
            position = torch.arange(0, width, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            twod_pe = torch.zeros(height, width, d_model*2)
            for xpos in range(height):
                for ypos in range(width):
                    twod_pe[xpos, ypos, :] = torch.cat([pe[0, xpos], pe[0, ypos]], dim=-1)

            twod_pe = twod_pe.flatten(0,1).T
            self.register_buffer('twod_pe', twod_pe)

    def forward(self, x):
        x = x + self.twod_pe[:, :x.size(-1)].unsqueeze(0)
        return self.dropout(x)
