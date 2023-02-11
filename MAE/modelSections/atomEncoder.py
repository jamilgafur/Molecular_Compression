import torch
import torch.nn.functional as F
from torch_geometric.nn.conv import  FastRGCNConv as RGCNConv

# Similar to https://github.com/pyg-team/pytorch_geometric/blob/master/examples/rgcn.py
class atomBondEncoderSection(torch.nn.Module):
    def __init__(self, atomClasses, bondClasses):
        super(atomBondEncoderSection, self).__init__()
        self.conv1 = RGCNConv(atomClasses, 64, bondClasses)
        self.conv2 = RGCNConv(64, 128,  bondClasses)
        self.conv3 = RGCNConv(128, 256,  bondClasses)

    def forward(self, atom, bond, connection):

        atom = torch.relu(self.conv1(atom, connection, bond))
        atom = F.dropout(atom,p=.5,training=self.training)

        atom = torch.relu(self.conv2(atom, connection, bond))
        atom = F.dropout(atom,p=.5,training=self.training)

        atombondEmbeddings = torch.relu(self.conv3(atom, connection, bond)) 
        

        return atombondEmbeddings     
