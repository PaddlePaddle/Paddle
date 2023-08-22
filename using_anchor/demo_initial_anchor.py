from symbol import parameters
import paddle
from paddle.autograd import PyLayer
from paddle.jit import to_static

# Inherit from PyLayer
class cus_tanh(PyLayer):
    @staticmethod
    def forward(ctx, x): 
        # ctx is a context object that store some objects for backward.
        y = paddle.tanh(x)       # <------ 仅仅包含 Paddle API 的计算
        # Pass tensors to backward.
        ctx.save_for_backward(y)
        return y

    @staticmethod
    # forward has only one output, so there is only one gradient in the input of backward.
    def backward(ctx, dy):
        # Get the tensors passed by forward.
        y, = ctx.saved_tensor()
        grad = dy * (1 - paddle.square(y))   # <------ 仅仅包含 Paddle API 的计算
        # forward has only one input, so only one gradient tensor is returned.
        return grad

class SimpleNet(paddle.nn.Layer):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.linear = paddle.nn.Linear(4, 8)
    
    @to_static
    def forward(self, x):
        y = self.linear(x)
        out = cus_tanh.apply(y)
        out = paddle.mean(out)
        return out


def train(net, opt):
    batch_num = 10
    for i in range(batch_num):
        data = paddle.randn([2, 4])
        out = net(data)
        out.backward()
        opt.step()
        opt.clear_grad()
        print("loss: ", out.item())
    
    save(net)


def save(net):
    path = "simple_net"
    x_spec = paddle.static.InputSpec(shape=[-1, 4], dtype='float32', name='x')
    paddle.jit.save(net, path, input_spec=[x_spec])


if __name__ == '__main__':
    net = SimpleNet()
    sgd = paddle.optimizer.SGD(0.001, parameters=net.parameters())
    train(net, sgd)