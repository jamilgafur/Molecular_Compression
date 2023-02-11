import torch
import numpy as np
import pandas as pd
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch_geometric.data import DataLoader


def readFiles(fileName="Data/zinc.csv", augment=3):
    # csv found here: https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv
    data = pd.read_csv(fileName)
    return data['smiles']

def normalizer(value,  listLength):
    return value/listLength

def plotData(train_losses, test_losses, test_accuracy):
    plt.plot(train_losses, label="training")
    plt.plot(test_losses, label="testing")
    plt.legend()
    plt.savefig("output/losses.png")
    plt.clf()
    plt.plot(test_accuracy, label="accuracy")
    plt.plot([1 for i in range(len(test_accuracy))], label="perfection")
    plt.legend()
    plt.savefig("output/test_accuracy.png")
    plt.clf()

def accuracy(predA, solutionA, predC, solutionC):
    atomAcc = 0
    for data in zip(predA, solutionA):
        if torch.argmax(data[0]) == data[1]:
            atomAcc+=1
    conAcc = 0
    for data in zip(predC, solutionC):
        if torch.argmax(data[0]) == torch.argmax(data[1]):
            conAcc+=1

    return atomAcc, conAcc

def processAtom(atom, atomLength):
    atom_matrix = np.zeros(shape=(len(atom),atomLength))
    for row, data in enumerate(atom):
        atom_matrix[row][data] = 1
    return torch.tensor(atom_matrix, dtype=torch.int32)

def trainModel(train_loader, test_loader, model, batch_size, epochs):
    savePath = "output/model.pt"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("\n\n============Training on: {}===========\n".format(device))
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.0005)
    atomLoss = nn.CrossEntropyLoss()
    connLoss = nn.MSELoss()
    model.train()

    epochTrainLoss = []
    atomAccuracyTest = []
    atomAccuracyTrain = []
    connAccuracy = []

    for epoch in range(epochs+1):
        totalTrainLoss = 0
        totalAtomAccuracyTrain = 0
        totalAtomAccuracyTest = 0
        totalConnAccuracy = 0

        # Train on batches
        for batch in train_loader:
            optimizer.zero_grad()
            batch = batch.to(device)
            hatAtom, hatConn = model(batch)
            Aloss = atomLoss(hatAtom, batch.atomposition)
            Closs = connLoss(hatConn, batch.connMatrix)
            loss = Aloss+Closs

            aac, conAcc = accuracy(hatAtom, batch.atomposition, hatConn, batch.connMatrix)
            totalAtomAccuracyTrain+=aac
            loss.backward()
            optimizer.step()
            totalTrainLoss+=loss.item()

        for batch in test_loader:
            batch = batch.to(device)
            hatAtom, hatConn = model(batch)
            aac, conAcc = accuracy(hatAtom, batch.atomposition, hatConn, batch.connMatrix)
            totalAtomAccuracyTest+=aac
            totalConnAccuracy+=conAcc

        totalTrainLoss/=len(train_loader)
        totalAtomAccuracyTest/=len(test_loader)
        totalAtomAccuracyTrain/=len(test_loader)
        totalConnAccuracy/=len(test_loader)

        print("Epoch:{}\ttrain:{}\tacc:{}\ttest AAcc:{}\tCAcc:{}".format(epoch, totalTrainLoss, totalAtomAccuracyTrain,totalAtomAccuracyTest, totalConnAccuracy))

        epochTrainLoss.append(totalTrainLoss)
        atomAccuracyTest.append(totalAtomAccuracyTest)
        atomAccuracyTrain.append(totalAtomAccuracyTrain)
        connAccuracy.append(totalConnAccuracy)
        plt.clf()
        plt.plot(epochTrainLoss)
        plt.savefig("output/TrainLoss.png")
        plt.clf()
        plt.plot(atomAccuracyTrain, label="train")
        plt.plot(atomAccuracyTest, label="test")
        plt.legend()
        plt.savefig("output/atomacc.png")
        plt.clf()
        

    