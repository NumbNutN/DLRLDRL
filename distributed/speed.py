import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

import sys
from pathlib import Path
import argparse
import deepspeed

sys.path.append(str(Path(__file__).parent.parent))
from tool.tool import *

def data_sinx(n, noise=0.1):
    x = torch.linspace(-1, 1, n).unsqueeze(1)  # (N,1) (N,1)
    y = torch.sin(x * torch.pi) + noise * torch.randn_like(x)
    return x, y

class R2RDataSet(Dataset):

    def __init__(self,sample_num):

        self.x, self.y = data_sinx(sample_num)
        self._len = sample_num
    
    def __len__(self):
        return self._len
    
    def __getitem__(self,idx):
        return self.x[idx], self.y[idx]
    

class UniversalApproximation(nn.Module):

    def __init__(self):
        super().__init__()

        self.hidden = nn.Linear(1,3)
        self.activation = nn.Tanh()
        self.output = nn.Linear(3,1)

    def forward(self,x):
        x= self.hidden(x)
        x = self.activation(x)
        return self.output(x)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="?")
    args = parser.parse_args()

    # dataset a map-style dataset
    dataSet = R2RDataSet(100)

    # dataloader a iterable dataset
    dataLoader = DataLoader(dataSet, batch_size=10, shuffle=True)

    # define the model
    model = UniversalApproximation()

    params = model.parameters()

    # initial deepspeed engine
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=params
    )

    # optimizer is not required
    # optimizer = optim.Adam(model.parameters(), lr=0.01)

    # define the MSE loss function
    criterion = nn.MSELoss()

    epoches = 1000
    for epoch in range(epoches):
        for i, (x,y_true) in enumerate(dataLoader):
            
            # forward using deepspeed method
            y_pred = model_engine(x)
            loss = criterion(y_pred,y_true)

            # backward
            # loss.backward()
            model_engine.backward(loss)

            # optimizer.step()
            model_engine.step()

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

    
    x = dataSet.x
    
    with torch.no_grad():
        hidden = model.activation(model.hidden(x)).detach()
    plot_hidden_units(x, hidden, "sinx")

    y_true = dataSet.y
    y_pred = model_engine(x)

    plot_predictions(x, y_pred, y_true)