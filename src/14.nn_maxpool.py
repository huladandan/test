# 最大池化的作用就是尽量减少数据维度，但有保留需要的数据特征,减小训练量

import torch
import torchvision.datasets
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("../dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64)

input_ = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]], dtype=torch.float32)
input_ = torch.reshape(input_, (-1, 1, 5, 5))
print(input_.shape)


class Hula(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=False)

    def forward(self, input):
        output = self.maxpool1(input)
        return output


hula = Hula()
# output = hula(input_)
# print(output)

writer = SummaryWriter("../logs_maxpool")
step = 0
for data in dataloader:
    imgs, target = data
    writer.add_images("input", imgs, step)
    output = hula(imgs)
    writer.add_images("output", output, step)
    step += 1

writer.close()
