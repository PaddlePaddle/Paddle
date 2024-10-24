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

import os
import shutil
import tempfile

import numpy as np

import paddle
from paddle.base import core
from paddle.base.executor import global_scope
from paddle.base.framework import IrGraph
from paddle.inference import Config, PrecisionType, create_predictor
from paddle.static.quantization import QuantizationTransformPassV2


class TestExplicitQuantizationModel:
    def setUp(self):
        paddle.enable_static()
        np.random.seed(1024)
        paddle.seed(1024)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(
            self.temp_dir.name, 'trt_explicit', self.__class__.__name__
        )

    def tearDown(self):
        shutil.rmtree(self.path)

    def build_program(self):
        train_prog = paddle.static.Program()
        with paddle.static.program_guard(train_prog):
            image = paddle.static.data(
                name='image', shape=[None, 1, 28, 28], dtype='float32'
            )
            label = paddle.static.data(
                name='label', shape=[None, 1], dtype='int64'
            )
            model = self.build_model()
            out = model.net(input=image, class_dim=10)
            cost = paddle.nn.functional.loss.cross_entropy(
                input=out, label=label
            )
            avg_cost = paddle.mean(x=cost)
            acc_top1 = paddle.metric.accuracy(input=out, label=label, k=1)
            optimizer = paddle.optimizer.Momentum(
                momentum=0.9,
                learning_rate=0.01,
                weight_decay=paddle.regularizer.L2Decay(4e-5),
            )
            optimizer.minimize(avg_cost)
        val_prog = train_prog.clone(for_test=True)
        place = paddle.CUDAPlace(0)
        exe = paddle.static.Executor(place)
        exe.run(paddle.static.default_startup_program())

        def transform(x):
            return np.reshape(x, [1, 28, 28]) - 127.5 / 127.5

        train_dataset = paddle.vision.datasets.MNIST(
            mode='train', backend='cv2', transform=transform
        )
        train_loader = paddle.io.DataLoader(
            train_dataset,
            places=place,
            feed_list=[image, label],
            drop_last=True,
            return_list=False,
            batch_size=64,
        )

        def train(program, stop_iter=128):
            for it, data in enumerate(train_loader):
                if it == 0:
                    self.input_data = data[0]['image']
                loss, top1 = exe.run(
                    program, feed=data, fetch_list=[avg_cost, acc_top1]
                )
                scope = global_scope()
                if it == stop_iter:
                    break

        train(train_prog)

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

        quant_train_prog = insert_qdq(train_prog, scope, place, for_test=False)
        quant_val_prog = insert_qdq(val_prog, scope, place, for_test=True)

        train(quant_train_prog)

        path_prefix = os.path.join(self.path, 'inference')
        paddle.static.save_inference_model(
            path_prefix, [image], [out], exe, program=quant_val_prog
        )

    def infer_program(self, trt_int8=False, collect_shape=False):
        config = Config(
            os.path.join(self.path, 'inference.pdmodel'),
            os.path.join(self.path, 'inference.pdiparams'),
        )
        config.enable_use_gpu(256, 0, PrecisionType.Float32)
        config.enable_memory_optim()
        if trt_int8:
            precision_mode = PrecisionType.Int8
        else:
            precision_mode = PrecisionType.Float32
        config.enable_tensorrt_engine(
            workspace_size=1 << 30,
            max_batch_size=1,
            min_subgraph_size=3,
            precision_mode=precision_mode,
            use_static=False,
            use_calib_mode=False,
        )
        if trt_int8:
            config.enable_tensorrt_explicit_quantization()
        shape_path = self.path + ".shape.txt"
        if collect_shape:
            config.collect_shape_range_info(shape_path)
        else:
            config.enable_tuned_tensorrt_dynamic_shape(shape_path)
        config.disable_glog_info()
        predictor = create_predictor(config)
        input_names = predictor.get_input_names()
        input_tensor = predictor.get_input_handle(input_names[0])
        input_tensor.reshape(self.input_data.shape())
        input_tensor.share_external_data(self.input_data)
        predictor.run()
        output_names = predictor.get_output_names()
        output_tensor = predictor.get_output_handle(output_names[0])
        output_data = output_tensor.copy_to_cpu()
        return output_data

    def test_model(self):
        with paddle.pir_utils.OldIrGuard():
            self.build_program()
            self.infer_program(trt_int8=False, collect_shape=True)
            baseline_output = self.infer_program(
                trt_int8=False, collect_shape=False
            )
            self.infer_program(trt_int8=True, collect_shape=True)
            trt_output = self.infer_program(trt_int8=True, collect_shape=False)
            trt_predict = np.argmax(trt_output, axis=1)
            baseline_predict = np.argmax(baseline_output, axis=1)
            same = (trt_predict == baseline_predict).sum() / len(trt_predict)
            self.assertGreaterEqual(
                same,
                0.9,
                "There are more then 10% output difference between int8 and float32 inference.",
            )
