import unittest
import paddle
from paddle.api.h import grid_sample 
class GridSampleTest(unittest.TestCase):
    def test_identity_matrix(self):
        input = paddle.to_tensor([1, 2, 3, 4]).reshape([1, 1, 2, 2])
        grid = paddle.to_tensor([[[[-1, -1], [1, 1]]]])
        output = paddle.zeros([1, 1, 2, 2])

        grid_sample(input, grid, output)

        expected_output = paddle.to_tensor([[[[1, 2], [3, 4]]]])
        self.assertTrue(paddle.allclose(expected_output, output))

if __name__ == '__main__':
    unittest.main()
