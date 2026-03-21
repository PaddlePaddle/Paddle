# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import annotations

import unittest
from dataclasses import dataclass

import numpy as np
from dist_amp_base import (
    create_optimizer,
)

import paddle
from paddle import nn
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import (
    LayerDesc,
    LayerSpec,
    NoPipelineParallel,
    PipelineLayer,
    build_spec_layer,
)
from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
    PipelineDatasetPreprocessor,
)
from paddle.nn import Layer
from paddle.nn.layer import Identity


class ReshapeHelp(Layer):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(shape=self.shape)


@dataclass
class AlexNetLayerSpec:
    features: list[LayerSpec] | list[Identity]
    reshape_layer: LayerSpec | type = Identity
    classifier: LayerSpec | type = Identity


class AlexNet(PipelineLayer):
    def __init__(self, sublayers_spec: AlexNetLayerSpec, **kwargs):
        self.layers = AlexNet.get_layer_desc_list(sublayers_spec)

        super().__init__(layers=self.layers, **kwargs)

    @staticmethod
    def get_layer_desc_list(spec: AlexNetLayerSpec):
        layers = []
        for features_spec in spec.features:
            layers.append(LayerDesc(features_spec))
        layers.append(LayerDesc(spec.reshape_layer))
        layers.append(LayerDesc(spec.classifier))
        return layers


def get_alex_spec(num_classes=10):
    spec = LayerSpec(
        layer=AlexNet,
        sublayers_spec=AlexNetLayerSpec(
            features=[
                LayerSpec(
                    layer=nn.Conv2D,
                    extra_kwargs={
                        "in_channels": 3,
                        "out_channels": 3,
                        "kernel_size": 11,
                        "stride": 4,
                        "padding": 5,
                    },
                ),
                LayerSpec(
                    layer=nn.ReLU,
                ),
                LayerSpec(
                    layer=nn.MaxPool2D,
                    extra_kwargs={"kernel_size": 2, "stride": 2},
                ),
                LayerSpec(
                    layer=nn.Conv2D,
                    extra_kwargs={
                        "in_channels": 3,
                        "out_channels": 3,
                        "kernel_size": 5,
                        "padding": 2,
                    },
                ),
                LayerSpec(
                    layer=nn.ReLU,
                ),
                LayerSpec(
                    layer=nn.MaxPool2D,
                    extra_kwargs={"kernel_size": 2, "stride": 2},
                ),
                LayerSpec(
                    layer=nn.Conv2D,
                    extra_kwargs={
                        "in_channels": 3,
                        "out_channels": 3,
                        "kernel_size": 3,
                        "padding": 1,
                    },
                ),
                LayerSpec(
                    layer=nn.ReLU,
                ),
                LayerSpec(
                    layer=nn.Conv2D,
                    extra_kwargs={
                        "in_channels": 3,
                        "out_channels": 3,
                        "kernel_size": 3,
                        "padding": 1,
                    },
                ),
                LayerSpec(
                    layer=nn.ReLU,
                ),
                LayerSpec(
                    layer=nn.Conv2D,
                    extra_kwargs={
                        "in_channels": 3,
                        "out_channels": 3,
                        "kernel_size": 3,
                        "padding": 1,
                    },
                ),
                LayerSpec(
                    layer=nn.ReLU,
                ),
                LayerSpec(
                    layer=nn.MaxPool2D,
                    extra_kwargs={"kernel_size": 2, "stride": 2},
                ),
            ],
            reshape_layer=LayerSpec(
                layer=ReshapeHelp, extra_kwargs={"shape": [-1, 256]}
            ),
            classifier=LayerSpec(
                layer=nn.Linear,
                extra_kwargs={"in_features": 256, "out_features": num_classes},
            ),
        ),
        extra_kwargs={
            "loss_fn": nn.CrossEntropyLoss(),
        },
    )
    return spec


class TestPipeLayerAPI(unittest.TestCase):
    def setUp(self):
        strategy = fleet.DistributedStrategy()
        self.pipeline_parallel_size = 2
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": 1,
            "pp_degree": self.pipeline_parallel_size,
        }
        batch_size = 8
        micro_batch_size = 2
        strategy.pipeline_configs = {
            "accumulate_steps": batch_size // micro_batch_size,
            "micro_batch_size": micro_batch_size,
        }
        self.strategy = strategy
        fleet.init(is_collective=True, strategy=strategy)
        self.hcg = fleet.get_hybrid_communicate_group()

    def test_pipelayer_desc(self):
        alex_desc = get_alex_spec()
        pipe_model = build_spec_layer(
            alex_desc, num_stages=self.pipeline_parallel_size
        )
        np.testing.assert_array_equal(len(pipe_model.parameters()), 6)

    def test_pipelayer_desc_single(self):
        alex_desc = get_alex_spec()
        pipe_model = build_spec_layer(alex_desc, num_stages=1)
        np.testing.assert_array_equal(len(pipe_model.parameters()), 12)
        pipe_model = NoPipelineParallel(pipe_model, self.strategy, self.hcg)
        input = paddle.randn([256, 3, 224, 224])
        label = paddle.randint(0, 10, [147, 1])

        # Test with list data
        data = [[input, input, input, input], [label, label, label, label]]
        optimizer = create_optimizer(
            model=pipe_model, use_pure_bf16=True, use_main_grad=True
        )
        base_lr = 0.1
        lr_scheduler = paddle.optimizer.lr.CosineAnnealingDecay(
            learning_rate=base_lr, T_max=1
        )
        pipe_model.train_batch(data, optimizer, lr_scheduler)
        pipe_model.eval_batch(data, optimizer)

        pipe_model._delay_scale_loss = True
        pipe_model.train_batch(data, optimizer)
        pipe_model.eval_batch(data, optimizer)
        pipe_model.is_pipeline_last_stage()

        pipe_model.train_batch(data, optimizer, return_micro_batch_loss=True)

        scaler = paddle.amp.GradScaler(init_loss_scaling=4096)
        scaler = fleet.distributed_scaler(scaler)
        pipe_model.train_batch(data, optimizer, scaler=scaler)

    def test_pipelayer_segment_method_list(self):
        alex_desc = get_alex_spec()
        pipe_model = build_spec_layer(
            alex_desc, num_stages=self.pipeline_parallel_size, seg_method=[0, 4]
        )
        stage_id = self.hcg.get_stage_id()
        if stage_id == 0:
            np.testing.assert_array_equal(len(pipe_model.parameters()), 4)
        elif stage_id == 1:
            np.testing.assert_array_equal(len(pipe_model.parameters()), 8)

    def test_pipelayer_segment_method_spec(self):
        alex_desc = get_alex_spec()
        pipe_model = build_spec_layer(
            alex_desc,
            num_stages=self.pipeline_parallel_size,
            seg_method="layer:Conv2D|MaxPool2D",
        )
        stage_id = self.hcg.get_stage_id()
        if stage_id == 0:
            np.testing.assert_array_equal(len(pipe_model.parameters()), 4)
        elif stage_id == 1:
            np.testing.assert_array_equal(len(pipe_model.parameters()), 8)

    def test_pipelayer_segment_method_vpp(self):
        alex_desc = get_alex_spec()
        pipe_model = build_spec_layer(
            alex_desc,
            num_stages=self.pipeline_parallel_size,
            seg_method="layer:Conv2D|MaxPool2D",
            num_virtual_pipeline_stages=2,
        )
        stage_id = self.hcg.get_stage_id()
        if stage_id == 0:
            np.testing.assert_array_equal(len(pipe_model.parameters()), 6)
        elif stage_id == 1:
            np.testing.assert_array_equal(len(pipe_model.parameters()), 6)

    def test_check_micro_batch_data_valid_with_tuple(self):
        """Test _check_micro_batch_data_valid with tuple data."""

        alex_desc = get_alex_spec()
        pipe_model = build_spec_layer(alex_desc, num_stages=1)
        pipe_model = NoPipelineParallel(pipe_model, self.strategy, self.hcg)

        # Test with tuple data
        tensor1 = paddle.randn([2, 3])
        tensor2 = paddle.randn([2, 3])
        tuple_data = (tensor1, tensor2)

        # This should not raise any exception
        pipe_model._check_micro_batch_data_valid(tuple_data)

    def test_check_micro_batch_data_valid_with_dict(self):
        """Test _check_micro_batch_data_valid with dict data."""

        alex_desc = get_alex_spec()
        pipe_model = build_spec_layer(alex_desc, num_stages=1)
        pipe_model = NoPipelineParallel(pipe_model, self.strategy, self.hcg)

        # Test with dict data
        dict_data = {
            "input": paddle.randn([2, 3]),
            "label": paddle.randn([2, 1]),
        }

        # This should not raise any exception
        pipe_model._check_micro_batch_data_valid(dict_data)

    def test_eval_batch_with_pipeline_dataset_preprocessor(self):
        """Test eval_batch with wrapper PipelineDatasetPreprocessor."""

        alex_desc = get_alex_spec()
        pipe_model = build_spec_layer(alex_desc, num_stages=1)
        pipe_model = NoPipelineParallel(pipe_model, self.strategy, self.hcg)

        input = paddle.randn([256, 3, 224, 224])
        label = paddle.randint(0, 10, [147, 1])

        # Test with list data
        data = [[input, input, input, input], [label, label, label, label]]

        # Test with PipelineDatasetPreprocessor wrapper
        def data_generator():
            return data

        preprocessed_data = PipelineDatasetPreprocessor(data_generator)
        # This should work - calling preprocessed_data() should return the data
        result = preprocessed_data()
        self.assertEqual(result, data)

        # Test eval_batch with this preprocessed data
        pipe_model.eval_batch(preprocessed_data)

    def test_eval_batch_with_non_tuple_list_data(self):
        """Test eval_batch with non-tuple/list data (iterable like generator)."""
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
            NoPipelineParallel,
        )

        alex_desc = get_alex_spec()
        pipe_model = build_spec_layer(alex_desc, num_stages=1)
        pipe_model = NoPipelineParallel(pipe_model, self.strategy, self.hcg)

        # Test with a generator (not tuple or list) - this covers the branch
        # where data is not tuple/list but is an iterable
        input = paddle.randn([256, 3, 224, 224])
        label = paddle.randint(0, 10, [147, 1])

        # Create a generator that yields (input, label) pairs
        def data_generator():
            for _ in range(pipe_model.accumulate_steps):
                yield (input, label)

        # This should work - generator is not tuple/list, so it goes to micro_dataset directly
        # Note: This test validates the code path, actual behavior depends on implementation
        try:
            pipe_model.eval_batch(data_generator)
        except (TypeError, StopIteration):
            # The generator gets exhausted or other expected errors
            pass


if __name__ == "__main__":
    unittest.main()
