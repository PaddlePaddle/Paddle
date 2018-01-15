import math
import unittest
from paddle.v2.fluid.distribute_transpiler import split_dense_variable
import paddle.v2.fluid as fluid
import paddle.v2.fluid.core as core
import random


class TestSplitVar(unittest.TestCase):
    def test_check_output(self):
        # split below shapes to 10 servers
        shapes = [[3, 5], [1024], [28, 784], [8, 1020], [800, 10]]
        expected_sizes = [
            [15], [1024],
            [2352, 2352, 2352, 2352, 2352, 2352, 2352, 2352, 2352, 784],
            [2040, 2040, 2040, 2040],
            [1150, 1150, 1150, 1150, 1150, 1150, 1100]
        ]
        var_list = []
        program = fluid.Program()
        for shape in shapes:
            var = program.global_block().create_var(
                name=str(random.randint(10000, 99999)),
                persistable=True,
                # dtype=core.VarDesc.VarType.LOD_TENSOR,
                shape=shape)
            var_list.append(var)
        blocks = split_dense_variable(var_list, 10)
        all_sizes = []
        for s in expected_sizes:
            for s2 in s:
                all_sizes.append(s2)
        for i, block_str in enumerate(blocks):
            varname, block_id, size = block_str.split(":")
            self.assertEqual(int(size), all_sizes[i])


if __name__ == '__main__':
    unittest.main()
