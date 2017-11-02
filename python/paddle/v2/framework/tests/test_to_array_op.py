import unittest
import numpy as np
import paddle.v2.framework.core as core
from paddle.v2.framework.op import Operator
from op_test import OpTest


class TestToArrayOp(unittest.TestCase):
    def check_with_place(self, place):
        scope = core.Scope()
        # input data
        x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        lod = [0, 3, 4, 10]
        rank_sort_table = [[4, 10, 2], [0, 3, 0], [3, 4, 1]]

        # get max_seq_len
        max_seq_len = rank_sort_table[0][1] - rank_sort_table[0][0]

        x_input = scope.var('X').get_tensor()
        x_input.set(np.asarray(x, dtype=np.float32), place)

        table_place = core.CPUPlace()
        table = scope.var('RankSortTable').get_tensor()
        table.set(np.asarray(rank_sort_table, dtype=np.int64), table_place)

        out = scope.var('Out')

        # create and run sgd operator
        to_array_op = Operator(
            "to_array", X='X', RankSortTable='RankSortTable', Out='Out')
        ctx = core.DeviceContext.create(place)
        to_array_op.run(scope, ctx)

        act_result = []
        for i in range(max_seq_len):
            act_result.append(core.get_fetch_variable(scope, "Out", i))

        act_result = map(lambda x: np.array(x), act_result)

        expect_result = [[4, 0, 3], [5, 1], [6, 2], [7], [8], [9]]
        expect_result = map(lambda x: np.asarray(x, dtype=np.float32), expect_result)

        self.assertEqual(len(act_result), 6)
        for i in range(max_seq_len):
            self.assertTrue(np.allclose(act_result[i], expect_result[i]))

    def test_to_array(self):
        places = [core.CPUPlace()]
        if core.is_compile_gpu():
            places.append(core.GPUPlace(0))
        for place in places:
            self.check_with_place(place)


if __name__ == "__main__":
    unittest.main()
