import unittest
import numpy as np
from op_test import OpTest


class TestConv2dOp(OpTest):
    def setUp(self):
        self.op_type = "conv2d"
        batch_size = 2
        input_channels = 3
        input_height = 5
        input_width = 5
        output_channels = 6
        filter_height = 3
        filter_width = 3
        stride = 1
        padding = 0
        output_height = (input_height - filter_height + 2 * padding
                         ) / stride + 1
        output_width = (input_width - filter_width + 2 * padding) / stride + 1
        input = np.random.random((batch_size, input_channels, input_height,
                                  input_width)).astype("float32")
        filter = np.random.random(
            (output_channels, input_channels, filter_height,
             filter_width)).astype("float32")
        output = np.ndarray(
            (batch_size, output_channels, output_height, output_width))

        for batchid in xrange(batch_size):
            for channelid in xrange(output_channels):
                for rowid in xrange(output_height):
                    for colid in xrange(output_width):
                        start_h = (rowid * stride) - padding
                        start_w = (colid * stride) - padding
                        output_value = 0.0
                        for inchannelid in xrange(input_channels):
                            for frowid in xrange(filter_height):
                                for fcolid in xrange(filter_width):
                                    input_value = 0.0
                                    inrowid = start_h + frowid
                                    incolid = start_w + fcolid
                                    if ((inrowid >= 0 and
                                         inrowid < input_height) and
                                        (incolid >= 0 and
                                         incolid < input_width)):
                                        input_value = input[batchid][
                                            inchannelid][inrowid][incolid]
                                    filter_value = filter[channelid][
                                        inchannelid][frowid][fcolid]
                                    output_value += input_value * filter_value
                        output[batchid][channelid][rowid][colid] = output_value

        self.inputs = {'Input': input, 'Filter': filter}
        self.outputs = {'Output': output}
        self.attrs = {'strides': [1, 1], 'paddings': [0, 0]}

    def test_check_output(self):
        self.check_output()


class TestConv2dGradOp(OpTest):
    def setUp(self):
        batch_size = 2
        input_channels = 3
        input_height = 5
        input_width = 5
        output_channels = 6
        filter_height = 3
        filter_width = 3
        stride = 1
        padding = 0
        output_height = (input_height - filter_height + 2 * padding
                         ) / stride + 1
        output_width = (input_width - filter_width + 2 * padding) / stride + 1
        input = np.random.random((batch_size, input_channels, input_height,
                                  input_width)).astype("float32")
        filter = np.random.random(
            (output_channels, input_channels, filter_height,
             filter_width)).astype("float32")

        self.op_type = 'conv2d'
        self.inputs = {'Input': input, 'Filter': filter}
        output = np.ndarray(
            (batch_size, output_channels, output_height, output_width))
        self.outputs = {'Output': output}
        self.attrs = {'strides': [1, 1], 'paddings': [0, 0]}

    #def test_compare_grad(self):
    #    self.compare_grad(self.op, self.inputs)

    def test_check_grad(self):
        self.check_grad(set(['Input', 'Filter']), 'Output')


if __name__ == '__main__':
    unittest.main()
