import sklearn.datasets.mldata
import sklearn.model_selection
import numpy
from config import DATA_HOME

__all__ = ['train_creator', 'test_creator']


def __mnist_reader_creator__(data, target):
    def reader():
        n_samples = data.shape[0]
        for i in xrange(n_samples):
            yield (data[i] / 255.0).astype(numpy.float32), int(target[i])

    return reader


TEST_SIZE = 10000
X_train = None
X_test = None
y_train = None
y_test = None


def __initialize_dataset__():
    global X_train, X_test, y_train, y_test
    if X_train is not None:
        return
    data = sklearn.datasets.mldata.fetch_mldata(
        "MNIST original", data_home=DATA_HOME)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        data.data, data.target, test_size=TEST_SIZE, random_state=0)


def train_creator():
    __initialize_dataset__()
    return __mnist_reader_creator__(X_train, y_train)


def test_creator():
    __initialize_dataset__()
    return __mnist_reader_creator__(X_test, y_test)


def unittest():
    assert len(list(test_creator()())) == TEST_SIZE


if __name__ == '__main__':
    unittest()
