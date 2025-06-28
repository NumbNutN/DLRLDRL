## Approximate Distribution p(z|x) by q_\theta(z)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist

import matplotlib.pyplot as plt
import numpy as np

class Net(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fc1 = nn.Linear(1, 3)
        self.fc2 = nn.Linear(3, 2)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        mu, std = self.fc2(x).chunk(2, dim=-1)
        return mu, std
    
# define z ~ p(z) = e^{-z}I_z(>=0)

def sample_data(n):
    # 人造数据服从 x ~ p(x|z) * p(z)
    z = dist.Exponential(1.0).sample((n,1))
    x = z + torch.rand_like(z)
    return x

def ELBO(x,q_z):
    """
    input: (batch,1)
    """

    # use Monte Carlo to approximate ELBO
    mu,sigma = q_z(x)

    # reparameter
    eps = torch.rand_like(mu)
    u = mu + sigma * eps

    z = torch.exp(u)

    log_px_z = dist.Normal(z,1).log_prob(x)

    log_pz = -z

    print("mu shaepe:", mu.shape," sigma shape:", sigma.shape, "u shape:", u.shape)

    log_qu_x = dist.Normal(mu,sigma).log_prob(u)
    log_qz_x = log_qu_x - u

    elbo = log_pz.squeeze(-1) + log_px_z.squeeze(-1) - log_qz_x.squeeze(-1)

    return elbo.mean()

# start train
def train(qz_x,num_epochs=1000,batch_size=64,lr=1e-3):
    
    optimizer = optim.Adam(qz_x.parameters(),lr=lr)
    for epoch in range(num_epochs):
        batch = sample_data(batch_size)
        loss = -ELBO(batch,qz_x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 200 == 0:
            print(f"Epoch {epoch:4d}  -ELBO = {loss.item():.4f}")
    

if __name__ == "__main__":
    qz_x = Net()
    train(qz_x)

    # visualize p(z,x) and q(z|x) given x = 1.5

    x = torch.tensor([[1.5]])
    mu, sigma = qz_x(x)
    z = np.linspace(0, 3, 100).reshape(-1, 1)
    p_z = np.exp(-z)  # p(z)
    p_x_z = dist.Normal(z, 1).log_prob(x).exp().numpy()  # p(x|z)
    p_zx = p_z * p_x_z  # p(z,x) = p(z) * p(x|z)
    q_z_x = dist.Normal(mu.item(), sigma.item()).log_prob(torch.tensor(z)).exp().numpy()  # q(z|x)
    plt.figure(figsize=(10, 6))
    plt.plot(z, p_zx, label='p(z,x)', color='blue')
    # plt.plot(z, p_x_z, label='p(x|z)', color='orange')
    plt.plot(z, q_z_x, label='q(z|x)', color='green')
    plt.title('Distributions')
    plt.xlabel('z')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid()
    plt.show()