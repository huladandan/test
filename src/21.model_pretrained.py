import torchvision

# train_data = torchvision.datasets.ImageNet("../data_image_net", split='train', dataloader=True,
#                                            transform=torchvision.transforms.ToTensor())
from torch import nn

vgg16_true = torchvision.models.vgg16()
vgg16_false = torchvision.models.vgg16()
dataset = torchvision.datasets.CIFAR10(root="../dataset", train=True, transform=torchvision.transforms.ToTensor(),
                                       download=True)

vgg16_true.classifier.add_module("add_linear", nn.Linear(1000, 10))
print(vgg16_true)

vgg16_false.classifier[6] = nn.Linear(4096, 10)
print(vgg16_false)
