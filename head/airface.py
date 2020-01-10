import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math


class AirFace(nn.Module):

    def __init__(self, in_features, out_features, device_id=None, s=64.0, m=0.40):
        super(AirFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device_id = device_id
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        if self.device_id == None:  # on cpu
            cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        else:                       # on gpu
            cosine = F.linear(F.normalize(input.cuda(self.device_id[0])),
                              F.normalize(self.weight.cuda(self.device_id[0])))

        # linear target logit
        cosine = torch.clamp(cosine, -1.0, 1.0)
        theta = torch.acos(cosine)
        theta_m = 1.0 - 2 * (theta + self.m) / math.pi
        theta = 1.0 - 2 * theta / math.pi

        # convert label to one-hot
        one_hot = torch.zeros(cosine.size())
        if self.device_id != None:
            one_hot = one_hot.cuda(self.device_id[0])
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        output = (one_hot * theta_m) + ((1.0 - one_hot) * theta)
        output *= self.s

        return output