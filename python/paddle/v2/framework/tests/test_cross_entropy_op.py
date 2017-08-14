import unittest
import numpy
from op_test_util import OpTestMeta
from gradient_checker import GradientChecker, create_op


class TestCrossEntropy(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        # TODO this unit test is not passed
        self.type = "onehot_cross_entropy"
        batch_size = 32
        class_num = 10
        X = numpy.random.random((batch_size, class_num)).astype("float32")
        label = 5 * numpy.ones(batch_size).astype("int32")
        self.inputs = {'X': X, 'label': label}
        Y = []
        for i in range(0, batch_size):
            Y.append(-numpy.log(X[i][label[i]]))
        self.outputs = {'Y': numpy.array(Y).astype("float32")}


class CrossEntropyGradOpTest(GradientChecker):
    def test_softmax_grad(self):
        op = create_op("onehot_cross_entropy")
        batch_size = 32
        class_num = 10
        inputs = {
            "X": numpy.random.uniform(
                0.1, 1.0, [batch_size, class_num]).astype("float32"),
            "label": (class_num / 2) * numpy.ones(batch_size).astype("int32")
        }
        self.check_grad(op, inputs, set("X"), "Y")


if __name__ == "__main__":
    unittest.main()
