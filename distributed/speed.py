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


def add_deepspeed_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="DeepSpeed")

    group.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training.")
    group.add_argument("--zero-stage", type=int, default=0, choices=[0, 1, 2, 3],
                       help="DeepSpeed ZeRO stage. 0: off, 1: offload optimizer, 2: offload parameters, "
                            "3: offload optimizer and parameters.")
    group.add_argument("--deepspeed_config",type=str,help="deepspeed optim")
    return parser


def setup_distributed_training(args):
    if args.local_rank == -1:
        # Not using distributed training
        return

    # Initialize deepspeed
    deepspeed.init_distributed()

    # Set the device for the current process
    torch.cuda.set_device(args.local_rank)
    print(f"Using device: {torch.cuda.current_device()}")

class R2RDataSet(Dataset):

    def __init__(self,sample_num):

        self.x, self.y = data_sinx(sample_num)
        self._len = sample_num
    
    def __len__(self):
        return self._len
    
    def __getitem__(self,idx):
        return self.x[idx].cuda(), self.y[idx].cuda()
    

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
    parser = add_deepspeed_args(parser)
    args = parser.parse_args()

    setup_distributed_training(args)

    # dataset a map-style dataset
    dataSet = R2RDataSet(100)

    # dataloader a iterable dataset
    dataLoader = DataLoader(dataSet, batch_size=10, shuffle=True)

    # define the model
    model = UniversalApproximation()

    params = model.parameters()

    # initial deepspeed engine
    # model parameter will be transfered to cuda:x
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

    
    x = dataSet.x.cuda()
    
    with torch.no_grad():
        hidden = model.activation(model.hidden(x)).detach()
    

    y_true = dataSet.y
    y_pred = model_engine(x)

    x = x.cpu()
    plot_hidden_units(x, hidden, "sinx")
    plot_predictions(x, y_pred, y_true)