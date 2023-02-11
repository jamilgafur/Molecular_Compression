import torch
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Dataset, Data
from .utils import processAtom
class SmileDataset(Dataset):
    def __init__(self, preprocessor, data):
        super(SmileDataset,  self).__init__()
        self.preprocessor = preprocessor
        self.data = self.processData(data)        
    
    def processData(self, data):
        values = []
        for smileString in tqdm(data):
            processed_output = self.preprocessor(smileString, train=False)
            atom = torch.tensor(processAtom(processed_output['atom'], self.preprocessor.atom_classes), dtype=torch.float32)
            bond = torch.tensor(processed_output['bond'], dtype=torch.long)
            conn = torch.tensor(processed_output['connectivity'].T,  dtype=torch.long)
            connMatrix = torch.tensor(self.connect_matrix(len(atom), processed_output['connectivity'].T), dtype=torch.float32)
            atomposition = torch.tensor(processed_output['atom'],dtype=torch.long)

            values.append(
                            Data(
                                x=atom, 
                                edge_attribute=bond, 
                                edge_index=conn, 
                                smile=smileString,
                                atomposition = atomposition,
                                connMatrix = connMatrix
                                )
                        )
        return values

    def connect_matrix(self,size,conn):
        conn_mat= np.zeros(shape=(size,size))
        for i in conn:
            conn_mat[i[0]][i[1]] = 1 
        
        return conn_mat 
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
        