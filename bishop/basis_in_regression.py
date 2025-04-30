import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def data_sinx(n, noise=0.1):
    x = torch.linspace(-1, 1, n).unsqueeze(1)
    y = torch.sin(x * torch.pi) + noise * torch.randn_like(x)
    return x, y

class UniversalApproximation(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.hidden = nn.Linear(1,3)
        self.activation = nn.Tanh()
        self.output = nn.Linear(3,1)

    def forward(self, x):
        x = self.activation(self.hidden(x))
        return self.output(x)
    

def plot_hidden_units(model, func_name):
    x = torch.linspace(-1, 1, 100).unsqueeze(1)
    with torch.no_grad():
        hidden = model.activation(model.hidden(x))
    
    plt.figure(figsize=(10, 5))
    for i in range(3):
        plt.plot(x.numpy(), hidden[:, i].numpy(), label=f"Unit {i+1}")
    
    plt.title(f"Hidden Layer Basis Functions ({func_name})")
    plt.xlabel("x"), plt.ylabel("Activation")
    plt.legend(), plt.grid(True)
    plt.show()


def plot_predictions(model, x, y):
    with torch.no_grad():
        y_pred = model(x)
    
    plt.figure(figsize=(10, 5))
    plt.scatter(x.numpy(), y.numpy(), label="Data", color='blue')
    plt.plot(x.numpy(), y_pred.numpy(), label="Model Prediction", color='red')
    plt.title("Model Predictions")
    plt.xlabel("x"), plt.ylabel("y")
    plt.legend(), plt.grid(True)
    plt.show()
    

x, y = data_sinx(100, 0.0)

model = UniversalApproximation()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()
    y_pred = model(x)
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')


plot_hidden_units(model, "sinx")

plot_predictions(model, x, y)

