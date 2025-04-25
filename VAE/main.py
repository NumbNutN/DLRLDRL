import torchvision
import torch
import torch.nn.functional as F
import numpy as np
import cv2

trainset = torchvision.datasets.MNIST('./data',train=True, download=True, transform=torchvision.transforms.ToTensor())
testset = torchvision.datasets.MNIST('./data',train=False, download=True, transform=torchvision.transforms.ToTensor())

xtrain = trainset.data.numpy()
ytrain = trainset.targets.numpy()
x_val_pre = testset.data[:1000].numpy()
y_val = testset.targets[:1000].numpy()


count = np.zeros(10)
idx = []
for i in range(0, len(ytrain)):
  for j in range(10):
    if(ytrain[i] == j):
      count[j] += 1
      if(count[j]<=1000):
        idx = np.append(idx, i)
        
y_train = ytrain[idx.astype('int')]
x_train_pre = xtrain[idx.astype('int')]

r,_,_ = x_train_pre.shape
x_train = np.zeros([r,14,14])
for i in range(r):
  a = cv2.resize(x_train_pre[i].astype('float32'), (14,14)) # Resizing the image from 28*28 to 14*14
  x_train[i] = a

r,_,_ = x_val_pre.shape
x_val = np.zeros([r,14,14])
for i in range(r):
  a = cv2.resize(x_val_pre[i].astype('float32'), (14,14)) # Resizing the image from 28*28 to 14*14
  x_val[i] = a


x_train = np.where(x_train > 128, 1, 0)
x_val = np.where(x_val > 128, 1, 0)
x_train = x_train.astype(np.float32)
x_val = x_val.astype(np.float32)

batch_size = 32
trainloader = torch.utils.data.DataLoader([[x_train[i], y_train[i]] for i in range(len(y_train))], shuffle=True, batch_size=batch_size)
testloader = torch.utils.data.DataLoader([[x_val[i], y_val[i]] for i in range(len(y_val))], shuffle=True, batch_size=100)

from model import VAE

idim = 196
hdim = 8
model = VAE(idim=idim,hdim=hdim)

# define the loss function

def loss_function(x,y,mu:torch.tensor,std:torch.tensor) -> tuple[torch.tensor, tuple[torch.tensor, torch.tensor]]:
    
    # KLD = 0.5*((std**2).sum() + (mu**2).sum() - hdim -torch.log(torch.prod(std**2)))
    KLD = 0.5* torch.sum(mu**2 + std**2 - 1 - torch.log(std**2))
    
    ERR = (((x-y)**2).sum())
    # ERR = F.binary_cross_entropy(y, x.view(-1, 196), reduction='sum')
    
    return ERR + KLD, (KLD, ERR)


def train(model, trainloader, optimizer, epoch):
   
   losses,kld_losses,likelihoods = [],[],[]

   for batch_idx, (data, target) in enumerate(trainloader):
        
        model.train()
        data = data.cuda()
        data = data.view(-1, 196)
        size = data.size(0)

        recon_batch, mu, std = model(data)
        loss, (KLD, likelihood) = loss_function(data, recon_batch, mu, std)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.item())
        kld_losses.append(KLD.item())
        likelihoods.append(likelihood.item())

        if batch_idx % 100 == 0:
        # if True:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tKLD: {:.6f}\tLikelihood: {:.6f}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                100. * batch_idx / len(trainloader), loss.item(), KLD.item(), likelihood.item()))
   return losses, kld_losses, likelihoods


import argparse
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    fig = plt.figure(figsize=(10, 8))
    # load parameter to choose train or eval
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--train', type=str,default='False', help='train the model')
    args = parser.parse_args()

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.train.lower() == 'true':
      # create model
      model = VAE(idim=idim, hdim=hdim).to(device)

      # optimizer and LR scheduler
      optimizer = torch.optim.Adam(model.parameters(), lr = 0.002)
      scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

      for epoch in range(1, 5):
          train(model, trainloader, optimizer, epoch)
          scheduler.step()

      # save model
      torch.save(model.state_dict(), 'vae.pth')
      print("Model saved to vae.pth")

    else:
      # load model
      model = VAE(idim=idim, hdim=hdim).to(device)
      model.load_state_dict(torch.load('vae.pth'))
      model.eval()

      # test the model
      with torch.no_grad():
          for batch_idx, (data, target) in enumerate(testloader):
              data = data.cuda()
              data = data.view(-1, 196)
              recon_batch, mu, std = model(data)
              loss, (KLD, likelihood) = loss_function(data, recon_batch, mu, std)
              print('Test Loss: {:.6f}\tKLD: {:.6f}\tLikelihood: {:.6f}'.format(loss.item(), KLD.item(), likelihood.item()))

          
          for i in range(8):
              data,t = next(iter(testloader))
              ori_img = fig.add_subplot(8,2,2*i+1)
              recon_img = fig.add_subplot(8,2,2*i+2)

              ori_img.imshow(data[0].cpu().numpy().reshape(14, 14))
              recon, mu, std = model(data.view(-1, 196).cuda())
              recon_img.imshow(recon[0].cpu().numpy().reshape(14, 14))

      plt.show()