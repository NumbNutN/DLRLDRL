import torch
from torch import nn

# define a Linear Layer (the instance of class MyLinear inherited from nn.Module) with nn.Parameter
# initialize the Linear Layer by `MyLinear(3,1)`
class MyLinear(nn.Module):
    def __init__(self,input,output):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(input,output))
        self.bias = nn.Parameter(torch.randn(output))

    def forward(self,input):
        return input @ self.weight + self.bias


intermediate = {}
# define a hook to print all the intemediate input
def get_inter(name):
    def hook(model,inputs,output:torch.Tensor):
        if 'activations' in name or name == '':
            return

        intermediate[name] = output.detach()    # exclude from gradient calculation
        print("hook trigger:",name,type(inputs[0]),type(output))
    return hook


def debug(model,input,output):
    # breakpoint()
    pass

# define a dynamic module with submodule
class DynamicNet(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.linears = nn.ModuleList([
            MyLinear(4,4) for _ in range(num_layers)
        ])

        self.activations = nn.ModuleDict({
            'relu':nn.ReLU(),
            'lrelu':nn.LeakyReLU()
        })

        self.final = MyLinear(4,1)

    def forward(self,x,act):
        for linear in self.linears:
            x = linear(x)
            #! activations has untrainable parameters
            x = self.activations[act](x)
        x= self.final(x)
        return x


net = DynamicNet(4)

# reigister hook to net
for name,layer in net.named_modules():
    layer.register_forward_hook(get_inter(name))

net.register_forward_hook(debug)


sample_input = torch.randn(4)
output = net(sample_input,'relu')

for param in net.named_parameters():
    print(param)

for name, inter in intermediate.items():
    print(name, inter)