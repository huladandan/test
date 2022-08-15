from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

img_path = "../dataset/train/ants/7759525_1363d24e88.jpg"
# img_path_abs = "W:\Desktop\pythonProject\test\dataset\train\ants\7759525_1363d24e88.jpg"
img = Image.open(img_path)

writer = SummaryWriter("logs")

# print(img)
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
# print(tensor_img)
writer.add_image("Tensor_img", tensor_img)
writer.close()