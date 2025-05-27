# a tutorial to use dataset and dataloader in pytorch
import torch

# a dataset with data and labels:
# [1,1] 1
# [1,2] 1
# [2,1] 1
# [2,2] 0

class MyDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.data = torch.tensor([[1, 1], [1, 2], [2, 1], [2, 2]], dtype=torch.float32)
        self.labels = torch.tensor([1, 1, 1, 0], dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    

# a dataloader with batch size 2
dataset = MyDataset()
dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

for epoch_idx, epoch in enumerate(range(10)):
    for i, batch in enumerate(dataloader):
        data, labels = batch
        print(f"Epoch {epoch}, Batch {i}:")
        print("Data:", data)
        print("Labels:", labels)
        # do something with data and labels
        # ...
    print(f"epoch {epoch_idx} finished")