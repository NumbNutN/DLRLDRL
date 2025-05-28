import matplotlib.pyplot as plt
import torch



def plot_hidden_units(x, hidden, func_name):
    
    '''
        x = torch.linspace(-1, 1, 100).unsqueeze(1)
        with torch.no_grad():
            hidden = model.activation(model.hidden(x)).detach()
    '''
    plt.figure(figsize=(10, 5))
    for i in range(3):
        plt.plot(x.numpy(), hidden[:, i].numpy(), label=f"Unit {i+1}")
    
    plt.title(f"Hidden Layer Basis Functions ({func_name})")
    plt.xlabel("x"), plt.ylabel("Activation")
    plt.legend(), plt.grid(True)
    plt.show()


def plot_predictions(x, y_pred, y_true):
    
    '''
        with torch.no_grad():
            y_pred = model(x)
    '''
    plt.figure(figsize=(10, 5))
    plt.scatter(x.numpy(), y_true.numpy(), label="Data", color='blue')
    plt.plot(x.numpy(), y_pred.numpy(), label="Model Prediction", color='red')
    plt.title("Model Predictions")
    plt.xlabel("x"), plt.ylabel("y")
    plt.legend(), plt.grid(True)
    plt.show()