import torch
import torchvision
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from torchvision import datasets, transforms
import numpy as np

from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser(description='Feature extraction and SVM training')
parser.add_argument("--using-svm", default='False', type=str,help="Use SVM for classification")
args = parser.parse_args()

model = torchvision.models.resnet18(pretrained=True).cuda()

torch.nn.Sequential(
    *list(model.children())[:-1],  # Remove the last layer
)

# for layer in list(model.children()):
#     print("layer type:", layer)  

# torch.nn.Flatten(),  # Flatten the output
# torch.nn.Linear(512, 10)  # Add a new linear layer for classification

for param in model.parameters():
    param.requires_grad = False  # Freeze the parameters of the pre-trained model

train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('.', train=True, download=True, transform=transforms.ToTensor()),batch_size=64, shuffle=True
)

print("generate representation for CIFAR")
model.eval()
features = []
labels = []
for inputs, target in tqdm(train_loader):
    inputs = inputs.cuda()
    with torch.no_grad():
        feature = model(inputs)
        features.append(feature.view(feature.size(0),-1).cpu().numpy())
        labels.append(target.cpu().numpy())
        


features = np.concatenate(features)
labels = np.concatenate(labels)

print("feature shape:",features.shape)

# Test
test_loader = torch.utils.data.DataLoader(datasets.CIFAR10('.', train=False, download=True, transform=transforms.ToTensor()), batch_size=64, shuffle=False)
test_features = []
test_labels = []

print("handle test data")
for inputs, target in tqdm(test_loader):
    inputs = inputs.cuda()
    with torch.no_grad():
        feature = model(inputs)
        test_features.append(feature.view(feature.size(0),-1).cpu().numpy())
        test_labels.append(target.cpu().numpy())

test_features = np.concatenate(test_features)
test_labels = np.concatenate(test_labels)


if args.using_svm.lower() == 'true':
    # using SVM
    print("training SVM")
    svm_model = SVC(kernel='linear',verbose=1)
    svm_model.fit(features, labels)

    # eval
    predictions = svm_model.predict(test_features)
    acc = accuracy_score(test_labels, predictions)
    print("Accuracy:", acc)


else:
    # using two layer MLP
    import torch.nn as nn
    import torch.optim as optim

    mlp = nn.Sequential(
        nn.Linear(features.shape[1], 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    ).cuda()

    opt = optim.Adam(mlp.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    train_data = torch.tensor(features, dtype=torch.float).cuda()
    train_labels_t = torch.tensor(labels, dtype=torch.long).cuda()
    dataset = torch.utils.data.TensorDataset(train_data, train_labels_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    for epoch in range(5):
        mlp.train()
        for x, y in loader:
            opt.zero_grad()
            out = mlp(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()

    test_data = torch.tensor(test_features, dtype=torch.float).cuda()
    test_labels_t = torch.tensor(test_labels, dtype=torch.long).cuda()
    mlp.eval()
    with torch.no_grad():
        preds = mlp(test_data)
        acc = (preds.argmax(dim=1) == test_labels_t).float().mean()
        print("Accuracy:", acc.item())