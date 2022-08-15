from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

writer = SummaryWriter("logs")
image_path = "../dataset/train/bees/16838648_415acd9e3f.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)

writer.add_image("test", img_array, 2, dataformats='HWC')
for i in range(100):
    writer.add_scalar("y = 2x", 2*i, i)

writer.close()
