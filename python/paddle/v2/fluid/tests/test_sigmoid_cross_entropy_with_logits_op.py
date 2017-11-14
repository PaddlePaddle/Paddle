import numpy as np
from op_test import OpTest
from scipy.special import logit
from scipy.special import expit


class TestSigmoidCrossEntropyWithLogitsOp1(OpTest):
    '''Test sigmoid_cross_entropy_with_logit_op with binary labels
    '''

    def setUp(self):
        self.op_type = "sigmoid_cross_entropy_with_logits"
        batch_size = 64
        num_classes = 20
        self.inputs = {
            'X': logit(
                np.random.uniform(0, 1, (batch_size, num_classes))
                .astype("float32")),
            'Labels': np.random.randint(0, 2, (batch_size, num_classes))
            .astype("float32")
        }

        # Fw Pass is implemented as elementwise sigmoid followed by
        # elementwise logistic loss
        # Labels * -log(sigmoid(X)) + (1 - labels) * -log(1 - sigmoid(X))
        sigmoid_X = expit(self.inputs['X'])
        term1 = self.inputs['Labels'] * np.log(sigmoid_X)
        term2 = (1 - self.inputs['Labels']) * np.log(1 - sigmoid_X)
        self.outputs = {'Out': -term1 - term2}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestSigmoidCrossEntropyWithLogitsOp2(OpTest):
    '''Test sigmoid_cross_entropy_with_logit_op with probabalistic labels
    '''

    def setUp(self):
        self.op_type = "sigmoid_cross_entropy_with_logits"
        batch_size = 64
        num_classes = 20
        self.inputs = {
            'X': logit(
                np.random.uniform(0, 1, (batch_size, num_classes))
                .astype("float32")),
            'Labels': np.random.uniform(0, 1, (batch_size, num_classes))
            .astype("float32")
        }

        # Fw Pass is implemented as elementwise sigmoid followed by
        # elementwise logistic loss
        # Labels * -log(sigmoid(X)) + (1 - labels) * -log(1 - sigmoid(X))
        sigmoid_X = expit(self.inputs['X'])
        term1 = self.inputs['Labels'] * np.log(sigmoid_X)
        term2 = (1 - self.inputs['Labels']) * np.log(1 - sigmoid_X)
        self.outputs = {'Out': -term1 - term2}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')
