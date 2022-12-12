import unittest
import numpy as np
import paddle.fluid as fluid
import paddle.nn.functional as functional
import paddle

class EmbeddingStatic(unittest.TestCase):
    def test_1(self):
        paddle.enable_static()
        prog = fluid.Program()
        with fluid.program_guard(prog):

            def test_bad_x():
                initializer = fluid.initializer.NumpyArrayInitializer(
                    np.random.random(size=(5, 3))
                )

                param_attr = fluid.ParamAttr(
                    name="emb_weight",
                    learning_rate=0.5,
                    initializer=initializer,
                    trainable=True,
                )

                weight = prog.global_block().create_parameter(
                    (5, 3), attr=param_attr, dtype="float32"
                )

                initializer_params = fluid.initializer.NumpyArrayInitializer(
                    np.random.random(size=(10,5))
                )

                param_attr_params = fluid.ParamAttr(
                    name = 'params',
                    learning_rate=0.5,
                    initializer=initializer_params,
                    trainable=True,
                )

                params = prog.global_block().create_parameter(
                    (10,5), attr = param_attr, dtype = "float32"
                )

                label = fluid.layers.data(
                    name="label",
                    shape=[5,3],
                    append_batch_size=False,
                    dtype="int64",
                )

                emb = functional.embedding_bag(
                    input = label, params = params, weight=weight, mode = "sum", name="embedding_bag"
                )

            test_bad_x()

    def test_2(self):
        paddle.enable_static()
        prog = fluid.Program()
        with fluid.program_guard(prog):

            def test_bad_x():
                initializer = fluid.initializer.NumpyArrayInitializer(
                    np.random.random(size=(5, 3))
                )

                param_attr = fluid.ParamAttr(
                    name="emb_weight",
                    learning_rate=0.5,
                    initializer=initializer,
                    trainable=True,
                )

                weight = prog.global_block().create_parameter(
                    (5, 3), attr=param_attr, dtype="float32"
                )

                initializer_params = fluid.initializer.NumpyArrayInitializer(
                    np.random.random(size=(10,5))
                )

                param_attr_params = fluid.ParamAttr(
                    name = 'params',
                    learning_rate=0.5,
                    initializer=initializer_params,
                    trainable=True,
                )

                params = prog.global_block().create_parameter(
                    (10,5), attr = param_attr, dtype = "float64"
                )

                label = fluid.layers.data(
                    name="label",
                    shape=[5,3],
                    append_batch_size=False,
                    dtype="int32",
                )

                emb = functional.embedding_bag(
                    input = label, params = params, weight=weight, mode = "sum", name="embedding_bag"
                )

            test_bad_x()


if __name__ == '__main__':
    unittest.main()