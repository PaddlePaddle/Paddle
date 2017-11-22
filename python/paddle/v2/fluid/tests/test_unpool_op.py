import unittest
import numpy as np
from op_test import OpTest


def unpool2dmax_forward_naive(input, indices, ksize, strides, paddings):
    s0, s1, s2, s3 = input.shape
    out_H=(s2 - 1) * strides[0] - 2 * paddings[0] + ksize[0]
    out_W=(s2 - 1) * strides[1] - 2 * paddings[1] + ksize[1]
    out = np.zeros((s0, s1, out_H, out_W))
    for nidx in xrange(s0):
        for cidx in xrange(s1):
            for h in xrange(s2):
                for w in xrange(s3):
                    index = indices[nidx, cidx, h, w]
                    hidx = (index - index % out_W) / out_W
                    widx = index % out_W
                    out[nidx, cidx, int(hidx), int(widx)] = input[nidx, cidx, h, w]

    return out


class TestUnpoolOp(OpTest):
    def setUp(self):
        self.op_type = "unpool"
        self.init_test_case()
        pre_input = np.random.random(self.shape).astype("float32")
        N, C, H, W = pre_input.shape
        H_out = (H - self.ksize[0] + 2 * self.paddings[0]) / self.strides[0] + 1
        W_out = (W - self.ksize[1] + 2 * self.paddings[1]) / self.strides[1] + 1
        input = np.zeros((N, C, H_out, W_out))
        indices = np.zeros((N, C, H_out, W_out))
        for i in xrange(H_out):
            for j in xrange(W_out):
                r_start = np.max((i * self.strides[0] - self.paddings[0], 0))
                r_end = np.min((i * self.strides[0] + self.ksize[0] - self.paddings[0], H))
                c_start = np.max((j * self.strides[1] - self.paddings[1], 0))
                c_end = np.min((j * self.strides[1] + self.ksize[1] - self.paddings[1], W))
                for nidx in xrange(N):
                    for cidx in xrange(C):
                        x_masked = pre_input[nidx, cidx, r_start:r_end, c_start:c_end]
                        input[nidx, cidx, i, j] = x_masked.max()
                        arg = x_masked.argmax()
                        indices[nidx, cidx, i, j] = (r_start + arg / self.ksize[1]) * W + c_start + arg % self.ksize[1]
        output = self.Unpool2d_forward_naive(input, indices, self.ksize, self.strides, self.paddings).astype("float32")
        self.inputs = {'X': input.astype('float32'),
                       'Y': indices.astype('int16')}
        self.attrs = {
                 'strides': self.strides,
                 'paddings': self.paddings,
                 'ksize': self.ksize,
                 'unpoolingtype': self.unpoolingtype,
                 }
        self.outputs = {'Out': output.astype('float32')}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', max_relative_error=0.5)

    def init_test_case(self):
        self.Unpool2d_forward_naive = unpool2dmax_forward_naive
        self.unpoolingtype = "max"
        self.shape = [6, 4, 5, 5]
        self.ksize = [3, 3]
        self.strides = [2, 2]
        self.paddings = [0, 0]



if __name__ == '__main__':
    unittest.main()
