import unittest
import paddle.v2.fluid as fluid
import numpy as np


def to_lodtensor(data, place):
    seq_lens = [len(seq) for seq in data]
    cur_len = 0
    lod = [cur_len]
    for l in seq_lens:
        cur_len += l
        lod.append(cur_len)
    flattened_data = np.concatenate(data, axis=0).astype("int64")
    flattened_data = flattened_data.reshape([len(flattened_data), 1])
    res = fluid.LoDTensor()
    res.set(flattened_data, place)
    res.set_lod([lod])
    return res


class TestDynamicRNN(unittest.TestCase):
    def setUp(self):
        self.place = fluid.CPUPlace()
        self.exe = fluid.Executor(self.place)

    # Input sequence data:
    # 1 2 3 4
    # 2 3 4
    # 5
    def test_simple_forward(self):
        in_data = fluid.layers.data("in_data", shape=[1], dtype="int64")
        seq = [[1, 2, 3, 4], [2, 3, 4], [5]]
        input_tensor = to_lodtensor(seq, self.place)
        rnn = fluid.layers.DynamicRNN()
        i = 2.0
        init = fluid.layers.fill_constant(shape=[3, 1], dtype="int64", value=0)
        with rnn.step():
            xt = rnn.step_input(in_data)
            mem = rnn.memory(init=init)
            scaled_xt = fluid.layers.scale(x=xt, scale=i)
            s = fluid.layers.elementwise_add(x=scaled_xt, y=mem)
            rnn.update_memory(s)
            rnn.output(s)
        out_data = rnn()

        expected_out = [[2], [6], [12], [20], [4], [10], [18], [10]]
        expected_lod = [[0, 4, 7, 8]]
        out = self.exe.run(feed={"in_data": input_tensor},
                           fetch_list=[out_data],
                           return_numpy=False)
        np.testing.assert_array_equal(np.array(out[0]), expected_out)
        self.assertEqual(out[0].lod(), expected_lod)


if __name__ == '__main__':
    unittest.main()
