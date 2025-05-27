import torch
import torch.nn as nn
import torch.nn.functional as F

# 设置种子便于复现
torch.manual_seed(0)

# 假设我们有一个两层 MLP
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# 创建网络 & 复制两个模型用于对比
net_A = SimpleNet()
net_B = SimpleNet()

# 复制模型权重（确保两者完全一致）
net_B.load_state_dict(net_A.state_dict())

# 创建一个 batch 的数据和对应标签
batch_input = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
batch_target = torch.tensor([[1.0], [0.0]])

# ---------- 方式 A: batch input ----------
optimizer_A = torch.optim.SGD(net_A.parameters(), lr=0.1)
optimizer_A.zero_grad()
output_A = net_A(batch_input)
loss_A = F.mse_loss(output_A, batch_target)
loss_A.backward()

# ---------- 方式 B: 平均 input ----------
mean_input = [batch_input.mean(dim=0, keepdim=True) for i in range(len(batch_input))]
mean_target = batch_target.mean(dim=0, keepdim=True)

optimizer_B = torch.optim.SGD(net_B.parameters(), lr=0.1)
optimizer_B.zero_grad()
output_B = net_B(mean_input)
loss_B = F.mse_loss(output_B, mean_target)
loss_B.backward()

# 比较结果
print("== 方式 A: Batch ==")
print("Output A:\n", output_A.detach())
print("Loss A:\n", loss_A.item())

print("\n== 方式 B: 平均输入 ==")
print("Output B:\n", output_B.detach())
print("Loss B:\n", loss_B.item())

# 比较第一层权重的梯度
print("\n== 梯度比较（第一层） ==")
print("Gradient A (fc1.weight):\n", net_A.fc1.weight.grad)
print("Gradient B (fc1.weight):\n", net_B.fc1.weight.grad)
