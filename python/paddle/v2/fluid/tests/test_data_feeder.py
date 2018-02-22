import paddle.v2.fluid as fluid


def test_converter():
    img = fluid.layers.data(name='image', shape=[1, 28, 28])
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    feeder = fluid.DataFeeder([img, label], fluid.CPUPlace())
    result = feeder.feed([[[0] * 784, [9]], [[1] * 784, [1]]])
    print(result)


if __name__ == '__main__':
    test_converter()
