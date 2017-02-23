import sklearn.datasets.mldata
import sklearn.model_selection
import numpy

__all__ = ['MNISTReader', 'train_reader_creator', 'test_reader_creator']

DATA_HOME = None


def __mnist_reader__(data, target):
    n_samples = data.shape[0]
    for i in xrange(n_samples):
        yield data[i].astype(numpy.float32), int(target[i])


class MNISTReader(object):
    """
    mnist dataset reader. The `train_reader` and `test_reader` method returns
    a iterator of each sample. Each sample is combined by 784-dim float and a
    one-dim label
    """

    def __init__(self, random_state):
        data = sklearn.datasets.mldata.fetch_mldata(
            "MNIST original", data_home=DATA_HOME)
        n_train = 60000
        self.X_train, self.X_test, self.y_train, self.y_test = sklearn.model_selection.train_test_split(
            data.data / 255.0,
            data.target.astype("int"),
            train_size=n_train,
            random_state=random_state)

    def train_reader(self):
        return __mnist_reader__(self.X_train, self.y_train)

    def test_reader(self):
        return __mnist_reader__(self.X_test, self.y_test)


__default_instance__ = MNISTReader(0)


def train_reader_creator():
    """
    Default train set reader creator.
    """
    return __default_instance__.train_reader


def test_reader_creator():
    """
    Default test set reader creator.
    """
    return __default_instance__.test_reader


def unittest():
    assert len(list(train_reader_creator()())) == 60000


if __name__ == '__main__':
    unittest()
