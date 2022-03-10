import paddle
import paddle.nn.functional as F
from paddle.nn import Layer
from paddle.vision.datasets import MNIST
from paddle.metric import Accuracy
from paddle.nn import Conv2D, MaxPool2D, Linear
from paddle.static import InputSpec
from paddle.jit import to_static
from paddle.vision.transforms import ToTensor


class AbsNet(paddle.nn.Layer):
    def __init__(self):
        super(AbsNet, self).__init__()

    def forward(self, x):
        x = paddle.abs(x)
        return x

if __name__ == '__main__':
    # paddle version
    print(paddle.__version__)
    # build network
    model = AbsNet()


    # save inferencing format model
    net = to_static(model,
                    input_spec=[InputSpec(shape=[None, 1, 28, 28], name='x')])
    paddle.jit.save(net, 'inference_model/lenet')
