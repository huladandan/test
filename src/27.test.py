import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn import Sequential, MaxPool2d, Conv2d, Linear, Flatten

image_path = "../image/dog.png"
image = Image.open(image_path)
image = image.convert('RGB')
print(image)
image = image.convert('RGB')
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])

image = transform(image)
print(image.shape)


class Hula(nn.Module):
    def __init__(self):
        super(Hula, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x


model = torch.load("hula_19.pth")
print(model)
image = torch.reshape(image, (1, 3, 32, 32))
model.eval()
with torch.no_grad():
    image = image.cuda()
    output = model(image)
print(output)

print(output.argmax(1))
