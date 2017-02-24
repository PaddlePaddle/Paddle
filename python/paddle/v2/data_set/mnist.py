import sklearn.datasets.mldata
import sklearn.model_selection
import numpy
from config import DATA_HOME

__all__ = ['MNIST', 'train_creator', 'test_creator']


def __mnist_reader_creator__(data, target):
    def reader():
        n_samples = data.shape[0]
        for i in xrange(n_samples):
            yield (data[i] / 255.0).astype(numpy.float32), int(target[i])

    return reader


class MNIST(object):
    """
    mnist dataset reader. The `train_reader` and `test_reader` method returns
    a iterator of each sample. Each sample is combined by 784-dim float and a
    one-dim label
    """

    def __init__(self, random_state=0, test_size=10000, **options):
        data = sklearn.datasets.mldata.fetch_mldata(
            "MNIST original", data_home=DATA_HOME)
        self.X_train, self.X_test, self.y_train, self.y_test = sklearn.model_selection.train_test_split(
            data.data,
            data.target,
            test_size=test_size,
            random_state=random_state,
            **options)

    def train_creator(self):
        return __mnist_reader_creator__(self.X_train, self.y_train)

    def test_creator(self):
        return __mnist_reader_creator__(self.X_test, self.y_test)


__default_instance__ = MNIST()
train_creator = __default_instance__.train_creator
test_creator = __default_instance__.test_creator


def unittest():
    size = 12045
    mnist = MNIST(test_size=size)
    assert len(list(mnist.test_creator()())) == size


if __name__ == '__main__':
    unittest()
