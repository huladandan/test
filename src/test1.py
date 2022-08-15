import torch

'''
x = torch.zeros([2, 3])
print(x.shape)
x = x.transpose(0, 1)
print(x.shape)
'''
'''
x = torch.zeros([2, 1, 3])
y = torch.zeros([2, 3, 3])
z = torch.zeros([2, 2, 3])
w = torch.cat([x, y, z], dim=1)
print(w.shape)
print(w.dtype)
print(torch.cuda.is_available())
'''
'''
x = torch.tensor([[1., 0.], [-1., 1.]], requires_grad=True)
z = x.pow(2).sum()
z.backward()
print(x.grad)
'''
"""
layer = torch.nn.Linear(32, 64)
print(layer.weight.shape)
print(layer.bias.shape)
"""


class A():
    def __call__(self, param):
        print('i can called like a function')
        print('传入参数的类型是：{}   值为： {}'.format(type(param), param))

        res = self.forward(param)
        return res

    def forward(self, input_):
        print('forward 函数被调用了')

        print('in  forward, 传入参数类型是：{}  值为: {}'.format(type(input_), input_))
        return input_


a = A()

input_param = a('i')
print("对象a传入的参数是：", input_param)



