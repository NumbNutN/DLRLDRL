import torch.nn as nn
import torch

class VAE(nn.Module):

    def __init__(self,idim,hdim):
        super(VAE,self).__init__()

        self.idim = idim
        self.hdim = hdim
        # encoder
        self.fc1 = nn.Linear(idim,128)
        self.fc21 = nn.Linear(128,hdim) #mu
        self.fc22 = nn.Linear(128,hdim) #sigma

        # decoder
        self.fc3 = nn.Linear(hdim,128)
        self.fc4 = nn.Linear(128,idim)

    def encoder(self,x):
        h = self.fc1(x)
        return self.fc21(h), self.fc22(h)

    def decoder(self,z):
        h = torch.tanh(self.fc3(z))
        return torch.sigmoid(self.fc4(h))

    def sample(self,mu,std):
        # get Gaussian Noise from random
        eps = torch.randn_like(std)
        return eps* std + mu
    
    def forward(self, x):
        mu, std = self.encoder(x)
        z = self.sample(mu,std)
        return self.decoder(z), mu, std

