import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn.conv import  SAGEConv
from torch_geometric.nn.glob import global_add_pool

from .modelSections.atomDecoder import atomBondDecoderSection 
from .modelSections.atomEncoder import atomBondEncoderSection 

class molModel(torch.nn.Module):
    def __init__(self, preprocessor):
        super(molModel, self).__init__()

        self.atomClasses = preprocessor.atom_classes
        self.bondClasses = preprocessor.bond_classes
        print("Atom class: {}".format(self.atomClasses))
        print("Bond class: {}".format(self.bondClasses))

        self.aes = atomBondEncoderSection(self.atomClasses, self.bondClasses)
        self.ads = atomBondDecoderSection(self.atomClasses)
    
    def forward(self, data):
        # Example passthrough:
        # Atom class: 47
        # Bond class: 67
        # Data(x=[30, 47], edge_index=[2, 66], edge_attribute=[66])
        # x = 30 atoms is set of 47 types
        # 66 edges 
        # each edge is a type from a set of 67
        atom = data.x

        bond = data.edge_attribute
        connection = data.edge_index

        # normalize data
        atom /= self.atomClasses        
        
        # Using the The relational graph attentional layer we create a feature matrix of 256 for each atom based on its neighbors and self
        atomEncoded = self.aes(atom, bond, connection)

        # we take this and reduce it down via Linear Layers back to the atomClass size (Should reconstruct the original atom matrix)
        atomDecoded = self.ads(atomEncoded)

        # We then generate the connectivity matrix and apply softmax (we just want the correct positions therefore it is a multiclass problem)
        connectionMatrix = torch.matmul(atomDecoded, atomDecoded.t())
        
        return atomDecoded.to(torch.float32), connectionMatrix