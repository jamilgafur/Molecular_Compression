import torch
import torch.nn.functional as F
from torch_geometric.nn.conv import  FastRGCNConv as RGCNConv

# Similar to https://github.com/pyg-team/pytorch_geometric/blob/master/examples/rgcn.py
class atomBondDecoderSection(torch.nn.Module):
    def __init__(self, atomClasses):
        super(atomBondDecoderSection, self).__init__()
        self.linear1 = torch.nn.Linear(256, 128)
        self.linear2 = torch.nn.Linear(128, 64)
        self.linear3 = torch.nn.Linear(64, 64)
        self.linear4 = torch.nn.Linear(64, atomClasses)
        

    def forward(self, atomEncoded):
        
        atomDecoded = F.relu(self.linear1(atomEncoded))
        atomDecoded = F.dropout(atomDecoded, p=0.25, training=self.training)
        atomDecoded = F.relu(self.linear2(atomDecoded))
        atomDecoded = F.dropout(atomDecoded, p=0.25, training=self.training)
        atomDecoded = F.relu(self.linear3(atomDecoded))
        atomDecoded = F.dropout(atomDecoded, p=0.25, training=self.training)
        atomDecoded = self.linear4(atomDecoded)


        return atomDecoded     
