# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved
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

import paddle
import paddle.nn.functional as F
from paddle.base import core
from paddle.base.executor import global_scope
from paddle.base.framework import IrGraph
from paddle.inference import Config, PrecisionType, create_predictor
from paddle.static.quantization import QuantizationTransformPassV2

paddle.enable_static()


class TestRemoveStrategyOpBase:
    def setUp(self):
        # Setup random seed
        np.random.seed(1024)
        paddle.seed(1024)

        # Initialize train dataset
        def transform(x):
            return np.reshape(x, [1, 28, 28]) - 127.5 / 127.5

        self.train_dataset = paddle.vision.datasets.MNIST(
            mode='train', backend='cv2', transform=transform
        )

    def build_model(self, data, label):
        conv2d = paddle.static.nn.conv2d(
            input=data, num_filters=6, filter_size=3
        )
        bn = paddle.static.nn.batch_norm(input=conv2d, act="relu")

        pool = F.max_pool2d(bn, kernel_size=2, stride=2)
        hidden = paddle.static.nn.fc(pool, size=10)
        cost = paddle.nn.functional.loss.cross_entropy(
            input=hidden, label=label
        )
        avg_cost = paddle.mean(x=cost)
        predict = paddle.argmax(hidden, axis=-1, dtype='int32')
        return avg_cost, predict

    def build_program(self):
        # This method builds the program and determine the following inference configuration
        self.serialized_program = None
        self.serialized_params = None
        self.input_data = None
        self.precision_mode = None
        self.dynamic_shape_info = None

    def train(self, program, feed_list, fetch_list, place, exe, stop_iter):
        train_loader = paddle.io.DataLoader(
            self.train_dataset,
            places=place,
            feed_list=feed_list,
            drop_last=True,
            return_list=False,
            batch_size=64,
        )
        for it, data in enumerate(train_loader):
            loss = exe.run(program, feed=data, fetch_list=fetch_list)
            if it == stop_iter:
                self.input_data = data[0]['X']
                break

    def infer_program(self, use_trt=False):
        config = Config()

        # Determine the predictor config
        config.set_model_buffer(
            self.serialized_program,
            len(self.serialized_program),
            self.serialized_params,
            len(self.serialized_params),
        )
        config.enable_use_gpu(256, 0, PrecisionType.Half)
        config.enable_memory_optim()
        # config.disable_glog_info()
        if use_trt:
            config.enable_tensorrt_engine(
                workspace_size=1 << 30,
                max_batch_size=128,
                min_subgraph_size=0,
                precision_mode=self.precision_mode,
                use_static=False,
                use_calib_mode=False,
            )
            config.set_trt_dynamic_shape_info(*self.dynamic_shape_info)
        predictor = create_predictor(config)

        # Set the input data
        input_names = predictor.get_input_names()
        input_tensor = predictor.get_input_handle(input_names[0])
        input_tensor.reshape(self.input_data.shape())
        input_tensor.share_external_data(self.input_data)

        predictor.run()

        # Return the output data
        output_names = predictor.get_output_names()
        output_tensor = predictor.get_output_handle(output_names[0])
        output_data = output_tensor.copy_to_cpu()
        return output_data

    def test_program(self):
        # 1. Build program and save the model and params as attributed serialized_program and serialized_params
        # 2. Run the inference with Paddle Inference
        # 3. Run the inference with Paddle-TRT
        # 4. Compare their predict label
        with paddle.pir_utils.OldIrGuard():
            self.build_program()
            baseline = self.infer_program()
            actual = self.infer_program(use_trt=True)
            same = (baseline == actual).sum() / len(baseline)
            self.assertGreaterEqual(
                same,
                0.9,
                "There are more then 10% output difference between Paddle-Inference and Paddle-TRT.",
            )


@unittest.skipIf(
    paddle.inference.get_trt_compile_version() < (8, 5, 1),
    "Quantization axis is consistent with Paddle after TRT 8.5.2.",
)
class TestRemoveStrategyOpAMP(TestRemoveStrategyOpBase, unittest.TestCase):
    def build_program(self):
        place = paddle.CUDAPlace(0)
        train_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        exe = paddle.static.Executor(place)

        # Build program
        with paddle.pir_utils.OldIrGuard():
            with paddle.static.program_guard(train_program, startup_program):
                data = paddle.static.data(
                    name='X', shape=[None, 1, 28, 28], dtype='float32'
                )
                label = paddle.static.data(
                    name='label', shape=[None, 1], dtype='int64'
                )
                avg_cost, predict = self.build_model(data, label)
                optimizer = paddle.optimizer.Momentum(learning_rate=0.01)
                optimizer = paddle.static.amp.decorate(
                    optimizer,
                    use_dynamic_loss_scaling=False,
                    use_pure_fp16=False,
                )
                optimizer.minimize(avg_cost)
        exe.run(startup_program)
        eval_program = train_program.clone(for_test=True)

        # Training
        self.train(
            train_program,
            feed_list=[data, label],
            fetch_list=[avg_cost],
            place=place,
            exe=exe,
            stop_iter=100,
        )

        # Save the inference configuration
        self.dynamic_shape_info = [
            {"X": (1, 1, 28, 28)},
            {"X": (128, 1, 28, 28)},
            {"X": (64, 1, 28, 28)},
        ]
        self.precision_mode = PrecisionType.Half
        self.serialized_program = paddle.static.serialize_program(
            [data], [predict], program=eval_program
        )
        self.serialized_params = paddle.static.serialize_persistables(
            [data], [predict], executor=exe, program=eval_program
        )


@unittest.skipIf(
    paddle.inference.get_trt_compile_version() < (8, 6, 1),
    "Quantization axis is consistent with Paddle after TRT 8.6.1.",
)
class TestRemoveStrategyOpAMPQAT(TestRemoveStrategyOpBase, unittest.TestCase):
    def build_program(self):
        place = paddle.CUDAPlace(0)
        train_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        exe = paddle.static.Executor(place)

        # Build program
        with paddle.pir_utils.OldIrGuard():
            with paddle.static.program_guard(train_program, startup_program):
                data = paddle.static.data(
                    name='X', shape=[None, 1, 28, 28], dtype='float32'
                )
                label = paddle.static.data(
                    name='label', shape=[None, 1], dtype='int64'
                )
                avg_cost, predict = self.build_model(data, label)
                optimizer = paddle.optimizer.Momentum(learning_rate=0.01)
                optimizer = paddle.static.amp.decorate(
                    optimizer,
                    use_dynamic_loss_scaling=False,
                    use_pure_fp16=False,
                )
                optimizer.minimize(avg_cost)
        exe.run(startup_program)
        eval_program = train_program.clone(for_test=True)

        # Training
        self.train(
            train_program,
            feed_list=[data, label],
            fetch_list=[avg_cost],
            place=place,
            exe=exe,
            stop_iter=100,
        )

        # Quantization aware training
        scope = global_scope()

        def insert_qdq(program, scope, place, for_test=False):
            graph = IrGraph(core.Graph(program.desc), for_test=for_test)
            transform_pass = QuantizationTransformPassV2(
                scope=scope,
                place=place,
                activation_quantize_type='moving_average_abs_max',
                weight_quantize_type='channel_wise_abs_max',
            )
            transform_pass.apply(graph)
            quant_program = graph.to_program()
            return quant_program

        quant_train_program = insert_qdq(
            train_program, scope, place, for_test=False
        )
        quant_eval_program = insert_qdq(
            eval_program, scope, place, for_test=True
        )
        self.train(
            quant_train_program,
            feed_list=[data, label],
            fetch_list=[avg_cost],
            place=place,
            exe=exe,
            stop_iter=100,
        )

        # Save the inference configuration
        self.dynamic_shape_info = [
            {"X": (1, 1, 28, 28)},
            {"X": (128, 1, 28, 28)},
            {"X": (64, 1, 28, 28)},
        ]
        self.precision_mode = PrecisionType.Int8
        self.serialized_program = paddle.static.serialize_program(
            [data], [predict], program=quant_eval_program
        )
        self.serialized_params = paddle.static.serialize_persistables(
            [data], [predict], executor=exe, program=quant_eval_program
        )


if __name__ == '__main__':
    unittest.main()
