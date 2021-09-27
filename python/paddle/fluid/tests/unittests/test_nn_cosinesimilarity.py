from __future__ import print_function

import unittest
import numpy as np
import paddle
import paddle.nn as nn
from paddle.static import Program, program_guard

def vector_1d_cos_sim(x1,x2):
    item1=0
    for i in range(len(x1)):
        item1=item1+x1[i]*x2[i]
    item2=((x1**2).sum())**(1/2)*((x2**2).sum())**(1/2)
    if item2==0: return 0
    else: return item1/item2

class TestCosineSimilarity(unittest.TestCase):
    def test_input_range(self):
        # x1/x2 should in range [10**-3, 10**9]
        cos_sim_func = nn.CosineSimilarity(axis=0)
        for i in range(-3,10):
            times=10**i
            m1 = np.random.rand(5)*times
            m2 = np.random.rand(5)*times
            x1 = paddle.to_tensor(m1).astype('float64')
            x2 = paddle.to_tensor(m2).astype('float64')
            self.assertAlmostEqual(cos_sim_func(x1,x2).numpy().tolist()[0],vector_1d_cos_sim(m1,m2))

    def test_input_dtype(self):
        # x1/x2 can be 'float32' or 'float64'
        cos_sim_func = nn.CosineSimilarity(axis=0)
        type_list=['float32','float64']
        for try_type in type_list:
            m1 = np.random.rand(2,3)
            m2 = np.random.rand(2,3)
            x1 = paddle.to_tensor(m1).astype(try_type)
            x2 = paddle.to_tensor(m2).astype(try_type)
            result = cos_sim_func(x1, x2)

    def test_errors(self):
        cos_sim_func = nn.CosineSimilarity(axis=0)
        m1 = np.random.rand(2,3)
        m2 = np.random.rand(2,3)

        # dtypes of x1 and x2 must be the 'float64' or 'float32'
        type_list=['bool','float16','int8','int16','int32','int64','uint8']
        for try_type in type_list:
            x1 = paddle.to_tensor(m1).astype(try_type)
            x2 = paddle.to_tensor(m2).astype(try_type)
            self.assertRaises(RuntimeError, cos_sim_func, x1, x2)

        # dtypes of x1 and x2 must be the same
        x1 = paddle.to_tensor(m1).astype('float64')
        x2 = paddle.to_tensor(m2).astype('float32')
        self.assertRaises(ValueError, cos_sim_func, x1, x2)

        # when x1 and x2 are larger than 10**9 or less than 10**-4, the result is error
        m1 = np.random.rand(5)
        m2 = np.random.rand(5)

        x1 = paddle.to_tensor(m1*10**10).astype('float64')
        x2 = paddle.to_tensor(m2*10**10).astype('float64')
        self.assertNotEqual(cos_sim_func(x1,x2).numpy().tolist()[0],vector_1d_cos_sim(m1,m2))

        x1 = paddle.to_tensor(m1*10**-4).astype('float64')
        x2 = paddle.to_tensor(m2*10**-4).astype('float64')
        self.assertNotEqual(cos_sim_func(x1,x2).numpy().tolist()[0],vector_1d_cos_sim(m1,m2))


if __name__ == '__main__':
    unittest.main()