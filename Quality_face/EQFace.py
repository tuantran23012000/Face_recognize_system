"""
Code for EQ Face Head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math


class EQFace(nn.Module):
    def __init__(self, in_features, out_features, s=64, m=0.5):
        super(EQFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, confidence, input, label, gaussian=True):
        weight = F.normalize(self.weight)
        cosine = F.linear(F.normalize(input), weight)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = phi.half()
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = torch.where(one_hot == 0, cosine, phi)
        if gaussian:
            confidence = torch.clamp(confidence - 0.2, 0, 1) * 1.2
            output = output * self.s * confidence
        else:
            output = output * self.s
        return output


class FaceQuality(nn.Module):
    """
    Structure (Pink Region): https://www.semanticscholar.org/paper/EQFace%3A-A-Simple-Explicit-Quality-Network-for-Face-Liu-Tan/4aa2b4a93165cba137554b4ba06992922c080fe4/figure/0

    Input:  Feature extracted by backbone
            Shape: [1, feature_dim]

    Output: tensor([[quality_score]])
            Shape: [1,1]

    Notes:  Change Sigmoid to Softmax to Normalize score in range 0->1
    """

    def __init__(self, feature_dim):
        super(FaceQuality, self).__init__()
        self.qualtiy= nn.Sequential(
            nn.Linear(feature_dim, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 2, bias=False),
            nn.Softmax(dim=1)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, feature):
        quality_score = self.qualtiy(feature)

        return quality_score[:, 0:1]
