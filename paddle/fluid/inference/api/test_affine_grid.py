import unittest
from paddle.api.h import affine_grid  
import numpy as np
import paddle


class TestAffineGrid(unittest.TestCase):
    
    def test_identity_matrix(self):
            # Test when the transformation matrix is an identity matrix
        batch_size = 2
        height = 4
        width = 4
        theta = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # 2x3 transformation matrix
        grid = affine_grid(theta, batch_size, height, width)
        expected_grid = ...  # Calculate the expected grid using numpy based on the identity transformation
        np.testing.assert_array_equal(grid.numpy(), expected_grid)

    def test_rotation(self):
        # Test when applying a rotation transformation
        batch_size = 1
        height = 3
        width = 3
        angle = np.pi / 2.0  # 90 degrees rotation
        theta = np.array([[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0]])  # 2x3 transformation matrix
        grid = affine_grid(theta, batch_size, height, width)
        expected_grid = ...  # Calculate the expected grid using numpy based on the rotation
        np.testing.assert_array_equal(grid.numpy(), expected_grid)

    # Add more test cases to cover different scenarios of your affine_grid function

if __name__ == '__main__':
    unittest.main()