import torch
from tqdm import tqdm
from nfp.preprocessing import SmilesPreprocessor
from nfp.preprocessing.features import atom_features_v1, bond_features_v1



from MAE.utils import readFiles, trainModel
from MAE.model import molModel
from MAE.smileData import SmileDataset

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    smileStrings = readFiles()[:2000]
    
    # split data into testing and training sets
    test_split = .3
    testSet    = smileStrings[:int(len(smileStrings)*test_split)]
    trainSet   = smileStrings[int(len(smileStrings)*test_split):]

    # Train the preprocessor
    preprocessor = SmilesPreprocessor(atom_features=atom_features_v1, bond_features=bond_features_v1,explicit_hs=False)
    for smiles in tqdm(trainSet):
        if not "ERROR" in smiles:
            preprocessor(smiles, train=True)
    
    # convert sets into DataSets
    print("Converting trainSet to graphs")
    trainSet = SmileDataset(preprocessor, trainSet)
    
    print("Converting testSet to graphs")
    testSet  = SmileDataset(preprocessor, testSet)

    # get the model
    model = molModel(preprocessor)
    print(model)

    # Detect GPU or CPU
    epochs = 100
    batch_size = 32
    learning_rate = 1e-3
    inputSize = 100

    # Compile the model
    optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

    trainModel(trainSet, testSet, model, batch_size, epochs)


main()