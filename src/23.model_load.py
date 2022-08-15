import torch
import torchvision

model1 = torch.load("vgg16_method1.pth")
print(model1)

model2 = torch.load("vgg16_method2.pth")
vgg16 = torchvision.models.vgg16()
vgg16.load_state_dict(model2)
print(model2)