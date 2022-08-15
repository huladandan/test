import torch
from torch.autograd import Variable
# import numpy as np

"""
print(torch.__version__)
np_date = np.arange(6).reshape((2,3))
torch_data = torch.from_numpy(np_date)
tensor2array = torch_data.numpy()

print(
    '\nnumpy', np_date,
    '\ntorch', torch_data,
    '\ntensor2array', tensor2array,
)
"""
"""
# abs
data = [-1, -2, 1, 2]
tensor = torch.FloatTensor(data)    # 32bit

print(
    '\nabs',
    '\nnumpy', np.abs(data),           # [1 2 1 2}
    '\ntorch', torch.abs(tensor),     # [1 2 1 2}
)
"""
tensor = torch.FloatTensor([[1, 2], [3, 4]])
variable = Variable(tensor)
print(tensor)
print(variable)

print(torch.cuda.is_available())
