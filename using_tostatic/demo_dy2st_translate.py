import paddle
from paddle.jit import to_static

class SimpleNet(paddle.nn.Layer):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.linear = paddle.nn.Linear(10, 3)

    @to_static
    def forward(self, x, y):
        out = self.my_fc(x)       # <---- self.other_func
        out = add_two(out, y)     # <---- other plain func
        return out

    def my_fc(self, x):
        out = self.linear(x)
        return out

# 此函数可以在任意文件
def add_two(x, y):
    out = x + y
    out = paddle.mean(out)
    return out

net = SimpleNet()
tempNet = to_static(add_two)
# 查看转写的代码内容
paddle.jit.set_code_level(100)

x = paddle.zeros([2,10], 'float32')
y = paddle.zeros([3], 'float32')

# out = net(x, y)
out = tempNet(x, x)
print("=== end ===")