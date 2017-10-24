import paddle.v2.framework.core as core
import paddle.v2.framework.framework as framework
import paddle.v2.framework.executor as executor

import numpy as np
import unittest
import os
import sys
import shutil

FOLDER_PATH = "./tmp_test_dir"


class TestSaveRestoreOp(unittest.TestCase):
    def test_save_restore_op(self):
        tensor_1_val = np.random.rand(3, 9).astype("float32")
        tensor_2_val = np.random.rand(4, 2).astype("float32")
        place = core.CPUPlace()

        program = framework.Program()
        block = program.global_block()
        v_a = block.create_var(
            dtype="float32", shape=[3, 9], lod_level=0, name="tensor_1")
        v_b = block.create_var(
            dtype="float32", shape=[4, 2], lod_level=0, name="tensor_2")

        t_1 = core.LoDTensor()
        t_1.set(tensor_1_val, place)
        t_2 = core.LoDTensor()
        t_2.set(tensor_2_val, place)
        block.append_op(
            type="save",
            inputs={"X": [v_a, v_b]},
            attrs={"folderPath": FOLDER_PATH})
        block.append_op(
            type="fill_constant",
            outputs={"Out": [v_a]},
            attrs={"shape": [2, 2],
                   "value": 0.0})
        block.append_op(
            type="fill_constant",
            outputs={"Out": [v_b]},
            attrs={"shape": [2, 2],
                   "value": 0.0})
        block.append_op(
            type="restore",
            outputs={"Out": [v_a, v_b]},
            attrs={"folderPath": FOLDER_PATH})

        if os.path.exists(FOLDER_PATH):
            shutil.rmtree(FOLDER_PATH)
        os.makedirs(FOLDER_PATH)

        exe = executor.Executor(place)
        out = exe.run(program,
                      feed={"tensor_1": t_1,
                            "tensor_2": t_2},
                      fetch_list=[v_a, v_b])

        self.assertTrue(os.path.isdir(FOLDER_PATH))
        self.assertTrue(os.path.isfile(FOLDER_PATH + "/__tensor_1__"))
        self.assertTrue(os.path.isfile(FOLDER_PATH + "/__tensor_2__"))

        self.assertTrue(np.array_equal(np.array(out[0]), tensor_1_val))
        self.assertTrue(np.array_equal(np.array(out[1]), tensor_2_val))

        shutil.rmtree(FOLDER_PATH)


if __name__ == "__main__":
    unittest.main()
