import unittest

import paddle
import numpy as np
from paddle.fluid.framework import _test_eager_guard

paddle.disable_static()


class EmbeddingBagDygraph(unittest.TestCase):
    def func_1(self):
        paddle.disable_static(paddle.CPUPlace())

        indices_data = np.random.randint(low = 0, high = 10, size = (3,2)).astype(np.int64)
        indices = paddle.to_tensor(indices_data, stop_gradient = False)

        weight_data = np.random.randint(low = 0, high = 10, size = (3,2)).astype(np.float32) 
        weight = paddle.to_tensor(weight_data, stop_gradient = False)

        embedding_bag = paddle.nn.EmbeddingBag(10, 3, mode = 'sum')

        w0 = np.full(shape=(10, 3), fill_value=2).astype(np.float32)
        embedding_bag._embedding.set_value(w0)

        adam = paddle.optimizer.Adam(
            parameters=[embedding_bag._embedding], learning_rate=0.01
        )
        adam.clear_grad()

        out = embedding_bag(input = indices, weight = weight)
        out.backward()
        adam.step()

    def test_1(self):
        with _test_eager_guard():
            self.func_1()
        self.func_1()

    def func_2(self):
        paddle.disable_static(paddle.CPUPlace())

        indices_data = np.random.randint(low = 0, high = 10, size = (3,2)).astype(np.int64)
        indices = paddle.to_tensor(indices_data, stop_gradient = False)

        weight_data = np.random.randint(low = 0, high = 10, size = (3,2)).astype(np.float32) 
        weight = paddle.to_tensor(weight_data, stop_gradient = False)

        embedding_bag = paddle.nn.EmbeddingBag(10, 3, mode = 'mean')

        w0 = np.full(shape=(10, 3), fill_value=2).astype(np.float32)
        embedding_bag._embedding.set_value(w0)

        adam = paddle.optimizer.Adam(
            parameters=[embedding_bag._embedding], learning_rate=0.01
        )
        adam.clear_grad()

        out = embedding_bag(input = indices, weight = None)
        out.backward()
        adam.step()

    def test_2(self):
        with _test_eager_guard():
            self.func_2()
        self.func_2()


if __name__ == '__main__':
    unittest.main()