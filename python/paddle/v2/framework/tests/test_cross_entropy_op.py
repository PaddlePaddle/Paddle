import unittest
import numpy
from op_test_util import OpTestMeta


class TestSGD(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        # TODO this unit test is not passed
        self.type = "onehot_cross_entropy"
        batch_size = 100
        class_num = 10
        X = numpy.random.random((batch_size, class_num)).astype("float32")
        label = 5 * numpy.ones(batch_size).astype("int32")
        self.inputs = {'X': X, 'label': label}
        Y = []
        for i in range(0, batch_size):
            Y.append(-numpy.log(X[i][label[i]]))
        self.outputs = {'Y': numpy.array(Y).astype("float32")}


if __name__ == "__main__":
    unittest.main()
