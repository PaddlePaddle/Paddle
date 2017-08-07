import unittest
import numpy
from op_test_util import OpTestMeta
from grad_test_util import GradChecker, create_op


class TestCrossEntropy(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = "onehot_cross_entropy"
        batch_size = 100
        class_num = 10
        self.X = numpy.random.random((batch_size, class_num)).astype("float32")
        self.label = 5 * numpy.ones(batch_size).astype("int32")
        Y = []
        for i in range(0, batch_size):
            Y.append(-numpy.log(self.X[i][self.label[i]]))
        self.Y = numpy.array(Y).astype("float32")


# TODO(superjom) add gradient check
class CrossEntropyGradOpTest(GradChecker):
    def test_softmax_grad(self):
        op = create_op("onehot_cross_entropy")
        batch_size = 100
        class_num = 10
        inputs = {
            "X": numpy.random.random((3, 4)).astype("float32"),
            "label": 5 * numpy.ones(batch_size).astype("int32")
        }
        self.assert_grad(op, inputs)


if __name__ == "__main__":
    unittest.main()
