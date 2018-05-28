#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import numpy as np
import math
from op_test import OpTest


def find_latest_set(num):
    return 1 + int(math.floor(math.log(num, 2)))


class CodeTable(object):
    def __init__(self, num_classes, code):
        self.c = num_classes + code

    def cal_index(self, bit):
        return (self.c >> (bit + 1)) - 1

    def get_length(self):
        return find_latest_set(self.c) - 1

    def cal_bit(self, bit):
        return self.c & (1 << bit)


def hsigmoid(x, w, ids, bias, num_classes):
    # code length = 
    # initialize pre out with dims={batch_size, code_length}
    print(
        "============================================= start ==========================================="
    )
    print("x:", x)
    print("w:", w)
    print("bias:", bias)
    print("ids:", ids)
    global pre_output
    batch_size = x.shape[0]
    code_length = find_latest_set(num_classes - 1)
    code_table = [0 for _ in range(code_length)]
    pre_output = np.zeros((batch_size, code_length))
    pre_sum = np.zeros((batch_size, 1))
    out = np.zeros((batch_size, 1)).astype("float32")
    # pre_out += code(bias)
    print("pre_output init: ", pre_output)
    for i in range(batch_size):
        code_table = CodeTable(num_classes, ids[i])
        length = code_table.get_length()
        print("code_path: ")
        for j in range(length):
            idx = code_table.cal_index(j)
            print("{}, \t".format(idx))
            pre_output[i][j] += bias[0][idx]
    print("pre_output after add bias: ", pre_output)
    # pre_out += code(w) * x
    # for i in range(batch_size):
    for j in range(batch_size):
        code_table = CodeTable(num_classes, ids[j])
        length = code_table.get_length()
        for k in range(length):
            idx = code_table.cal_index(k)
            sum = 0.0
            for l in range(x.shape[1]):
                sum += w[idx][l] * x[j][l]
            pre_output[j][k] += sum
    print("pre_output after mul w,x: ", pre_output)
    # clip[-40.0, 40.0]
    np.clip(pre_output, -40.0, 40.0)
    # out(i, 0) = \sum_j  bit(i, j) * preout(i, j)
    for i in range(batch_size):
        code_table = CodeTable(num_classes, ids[i])
        length = code_table.get_length()
        sum = 0.0
        for j in range(length):
            if code_table.cal_bit(j):
                sum += pre_output[i][j]
        out[i] = -1.0 * sum
    print("out after sum: ", out)
    # soft relu
    np.clip(pre_output, -40.0, 40.0)
    pre_output = np.log(1 + np.exp(pre_output))
    print("pre_output after softrelu: ", pre_output)
    pre_sum = pre_output.sum(1).reshape((batch_size, 1))
    print("pre_sum: ", pre_sum)
    out += pre_sum
    print("out finally: ", out)
    print(
        "=========================================== end ============================================="
    )
    return out


class TestHSigmoidOp(OpTest):
    def setUp(self):
        self.op_type = "hierarchical_sigmoid"
        num_classes = 4
        embded_size = 1
        batch_size = 1
        x = np.random.random((batch_size, embded_size)).astype("float32")
        w = np.random.random((num_classes - 1, embded_size)).astype("float32")
        ids = np.random.randint(0, num_classes, batch_size)
        bias = np.random.random((1, num_classes - 1)).astype("float32")
        self.attrs = {'num_classes': num_classes}
        self.inputs = {'X': x, 'W': w, 'Ids': ids, 'Bias': bias}
        # bias[0][0] -= 0.005
        out = hsigmoid(x, w, ids, bias, num_classes)
        # bias[0][0] += 0.01
        # out2 = hsigmoid(x, w, ids, bias, num_classes)
        # delta = out.sum(0).mean() - out2.sum(0).mean()
        # print("delta: ", delta)
        self.outputs = {'PreOut': pre_output, 'Out': out}

#    def test_check_output(self):
#        self.check_output()

    def test_check_grad(self):
        self.check_grad(['Bias', 'X', 'W'], 'Out', no_grad_set=set('Ids'))

if __name__ == '__main__':
    unittest.main()
