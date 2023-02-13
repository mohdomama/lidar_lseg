from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from lidar_fields.util import get_embedder

# from gridencoder import GridEncoder


class LidarLsegModel(nn.Module):
    def __init__(self) -> None:
        super(LidarLsegModel, self).__init__()
        self.pos_embedder, _ = get_embedder(3)

        self.linear = nn.Sequential(
            nn.Linear(21, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256,512)
        )


    def forward(self, x):
        x = self.pos_embedder(x)
        out = self.linear(x)
        return out
