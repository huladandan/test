import torch
from torch import nn


class MyModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input_):
        output = input_ + 1
        return output


my_module = MyModule()
x = torch.tensor(1.0)
output = my_module(x)
print(output)
