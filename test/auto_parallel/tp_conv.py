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


import random

import numpy as np

import paddle
import paddle.distributed as dist
from paddle import nn

dist.init_parallel_env()


class SimpleConvNet(nn.Layer):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        padding,
        bias_attr,
        stride,
        data_format="NCHW",
    ):
        super().__init__()
        self.conv1 = nn.Conv2D(
            in_channel,
            out_channel,
            kernel_size=kernel_size,
            padding=padding,
            data_format=data_format,
            bias_attr=bias_attr,
            stride=stride,
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        return self.relu(x)


class TestTPConv:
    def __init__(self):
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self._tp_mesh = dist.ProcessMesh(
            list(range(self.world_size)), dim_names=["tp"]
        )

    def set_seed(self, seed):
        paddle.seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def _test_intermediate(
        self,
        N,
        C,
        H,
        W,
        kernel_size,
        padding,
        bias_attr,
        mesh,
        test_name="conv_test",
        dtype_str="float32",
        data_format="NCHW",
        stride=1,
    ):
        self.set_seed(2025)

        dist.auto_parallel.set_mesh(mesh)

        conv_layer = SimpleConvNet(
            C,
            C,
            kernel_size=kernel_size,
            padding=padding,
            data_format=data_format,
            bias_attr=bias_attr,
            stride=stride,
        )
        original_weight = conv_layer.conv1.weight
        conv_layer.conv1.weight = original_weight

        if data_format == "NCHW":
            input_tensor = paddle.randn([N, C, H, W])
            shard_axis_input = 3
        else:
            input_tensor = paddle.randn([N, H, W, C])
            shard_axis_input = 2

        input_placements = [
            dist.Replicate() for _ in range(len(mesh.dim_names))
        ]
        mp_dim_index = mesh.dim_names.index("mp")
        input_placements[mp_dim_index] = dist.Shard(shard_axis_input)

        sharded_input = dist.shard_tensor(input_tensor, mesh, input_placements)
        output_ref = conv_layer(input_tensor)
        loss_ref = output_ref.mean()
        loss_ref.backward()
        weight_grad_ref = conv_layer.conv1.weight.grad.clone()

        if (
            conv_layer.conv1.bias is not None
            and conv_layer.conv1.bias.grad is not None
        ):
            bias_grad_ref = conv_layer.conv1.bias.grad.clone()

        conv_layer.clear_gradients()
        conv_layer.conv1.weight = original_weight

        opt = paddle.optimizer.AdamW(
            learning_rate=0.001, parameters=conv_layer.parameters()
        )
        mp_config = {"parallelize_plan": {"conv1": dist.ConvParallel()}}

        parallel_config = {
            "mp_config": mp_config,
        }

        dist_model, dist_opt = dist.parallelize(
            conv_layer, opt, config=parallel_config
        )

        output_intermediate = dist_model(input_tensor)
        loss_intermediate = paddle.mean(output_intermediate)
        loss_intermediate.backward()
        weight_grad_intermediate = dist_model.conv1.weight.grad.clone()

        if (
            dist_model.conv1.bias is not None
            and dist_model.conv1.bias.grad is not None
        ):
            bias_grad_intermediate = dist_model.conv1.bias.grad.clone()

        def compare_tensors(name, tensor1, tensor2):
            np.testing.assert_allclose(
                tensor1.numpy(), tensor2.numpy(), rtol=1e-8, atol=1e-8
            )

        def compare_grads(name, grad1, grad2):
            np.testing.assert_allclose(
                grad1.numpy(), grad2.numpy(), rtol=1e-6, atol=1e-6
            )

        if data_format == "NCHW":
            w_size = output_ref.shape[-1]
        else:
            w_size = output_ref.shape[-2]

        if dist.get_rank() == 0:
            start_index = 0
            end_index = w_size // 2
        else:
            start_index = w_size // 2
            end_index = w_size

        if data_format == "NCHW":
            compare_tensors(
                "output",
                output_ref[:, :, :, start_index:end_index],
                output_intermediate._local_value(),
            )
        else:
            compare_tensors(
                "output",
                output_ref[:, :, start_index:end_index, :],
                output_intermediate._local_value(),
            )

        compare_grads("w", weight_grad_ref, weight_grad_intermediate)

        if (
            conv_layer.conv1.bias is not None
            and conv_layer.conv1.bias.grad is not None
        ):
            compare_grads("b", bias_grad_ref, bias_grad_intermediate)

    def _test_conv_case(
        self,
        N,
        C,
        H,
        W,
        kernel_size,
        padding,
        bias_attr,
        mesh,
        test_name="conv_test",
        dtype_str="float32",
        data_format="NCHW",
        stride=1,
    ):
        self.set_seed(2025)

        conv_layer = nn.Conv2D(
            C,
            C,
            kernel_size=kernel_size,
            padding=padding,
            bias_attr=bias_attr,
            data_format=data_format,
            stride=stride,
        )
        original_weight = conv_layer.weight
        conv_layer.weight = original_weight

        if data_format == "NCHW":
            input_tensor = paddle.randn([N, C, H, W])
            shard_axis_input = 3
        else:
            input_tensor = paddle.randn([N, H, W, C])
            shard_axis_input = 2

        output_ref = conv_layer(input_tensor)
        loss_ref = output_ref.mean()
        loss_ref.backward()
        weight_grad_ref = conv_layer.weight.grad.clone()

        if conv_layer.bias is not None and conv_layer.bias.grad is not None:
            bias_grad_ref = conv_layer.bias.grad.clone()

        conv_layer.clear_gradients()
        conv_layer.weight = original_weight

        rank = dist.get_rank()
        input_placements = [
            dist.Replicate() for _ in range(len(mesh.dim_names))
        ]
        mp_dim_index = mesh.dim_names.index("mp")
        input_placements[mp_dim_index] = dist.Shard(shard_axis_input)

        sharded_input = dist.shard_tensor(input_tensor, mesh, input_placements)

        output_sharded = conv_layer(sharded_input)
        loss_sharded = paddle.mean(output_sharded)
        loss_sharded.backward()

        weight_grad_shard = conv_layer.weight.grad.clone()
        if conv_layer.bias is not None and conv_layer.bias.grad is not None:
            bias_grad_shard = conv_layer.bias.grad.clone()

        def compare_grads(name, grad1, grad2):
            np.testing.assert_allclose(
                grad1.numpy(), grad2.numpy(), rtol=1e-6, atol=1e-7
            )

        def compare_tensors(name, tensor1, tensor2):
            np.testing.assert_allclose(
                tensor1.numpy(), tensor2.numpy(), rtol=1e-8, atol=1e-8
            )

        if data_format == "NCHW":
            w_size = output_ref.shape[-1]
        else:
            w_size = output_ref.shape[-2]

        if dist.get_rank() == 0:
            start_index = 0
            end_index = w_size // 2
        else:
            start_index = w_size // 2
            end_index = w_size

        if data_format == "NCHW":
            compare_tensors(
                "output",
                output_ref[:, :, :, start_index:end_index],
                output_sharded._local_value(),
            )
        else:
            compare_tensors(
                "output",
                output_ref[:, :, start_index:end_index, :],
                output_sharded._local_value(),
            )

        compare_grads("w", weight_grad_ref, weight_grad_shard)

        if conv_layer.bias is not None and conv_layer.bias.grad is not None:
            compare_grads("b", bias_grad_ref, bias_grad_shard)

    def run_test_cases(self):
        mesh1 = dist.ProcessMesh([0, 1], dim_names=['mp'])

        # ========= Case 1: padding > 0, stride = 1 =========
        # Typical convolution with halo exchange required.
        self._test_conv_case(
            N=1,
            C=10,
            H=32,
            W=32,
            kernel_size=3,
            padding=1,
            bias_attr=True,
            mesh=mesh1,
        )
        self._test_conv_case(
            N=2,
            C=8,
            H=16,
            W=32,
            kernel_size=5,
            padding=2,
            bias_attr=False,
            mesh=mesh1,
        )
        self._test_conv_case(
            N=4,
            C=6,
            H=28,
            W=28,
            kernel_size=3,
            padding=1,
            bias_attr=True,
            mesh=mesh1,
        )

        # NHWC format with padding > 0
        self._test_conv_case(
            N=2,
            C=8,
            H=16,
            W=32,
            kernel_size=3,
            padding=1,
            bias_attr=True,
            mesh=mesh1,
            data_format="NHWC",
        )
        self._test_conv_case(
            N=4,
            C=6,
            H=28,
            W=28,
            kernel_size=5,
            padding=2,
            bias_attr=False,
            mesh=mesh1,
            data_format="NHWC",
        )

        # ========= Case 2: padding = 0, stride == kernel_size =========
        # No halo exchange needed, input width must be divisible by stride.
        self._test_conv_case(
            N=1,
            C=10,
            H=32,
            W=32,
            kernel_size=1,
            padding=0,
            bias_attr=True,
            mesh=mesh1,
            stride=1,
        )
        self._test_conv_case(
            N=4,
            C=6,
            H=32,
            W=32,
            kernel_size=2,
            padding=0,
            bias_attr=False,
            mesh=mesh1,
            stride=2,
        )
        self._test_conv_case(
            N=2,
            C=8,
            H=16,
            W=32,
            kernel_size=4,
            padding=0,
            bias_attr=True,
            mesh=mesh1,
            stride=4,
        )

        # NHWC format with padding = 0
        self._test_conv_case(
            N=1,
            C=10,
            H=32,
            W=32,
            kernel_size=2,
            padding=0,
            bias_attr=True,
            mesh=mesh1,
            stride=2,
            data_format="NHWC",
        )
        self._test_conv_case(
            N=4,
            C=6,
            H=32,
            W=32,
            kernel_size=4,
            padding=0,
            bias_attr=False,
            mesh=mesh1,
            stride=4,
            data_format="NHWC",
        )

        # ========= Case 3: 2D ProcessMesh (dp + tp) =========
        mesh2 = dist.ProcessMesh([[0, 1]], dim_names=['dp', 'mp'])

        # padding > 0
        self._test_conv_case(
            N=2,
            C=8,
            H=32,
            W=32,
            kernel_size=3,
            padding=1,
            bias_attr=True,
            mesh=mesh2,
        )
        self._test_conv_case(
            N=4,
            C=6,
            H=28,
            W=28,
            kernel_size=5,
            padding=2,
            bias_attr=False,
            mesh=mesh2,
        )

        # padding = 0
        self._test_conv_case(
            N=2,
            C=8,
            H=16,
            W=32,
            kernel_size=1,
            padding=0,
            bias_attr=True,
            mesh=mesh2,
            stride=1,
        )

        # NHWC format, both padding > 0 and = 0
        self._test_conv_case(
            N=4,
            C=6,
            H=28,
            W=28,
            kernel_size=3,
            padding=1,
            bias_attr=True,
            mesh=mesh2,
            data_format="NHWC",
        )
        self._test_conv_case(
            N=1,
            C=10,
            H=32,
            W=32,
            kernel_size=4,
            padding=0,
            bias_attr=True,
            mesh=mesh2,
            stride=4,
            data_format="NHWC",
        )

        self._test_intermediate(
            N=1,
            C=10,
            H=32,
            W=32,
            kernel_size=3,
            padding=1,
            bias_attr=True,
            mesh=mesh1,
        )
        self._test_intermediate(
            N=1,
            C=10,
            H=32,
            W=32,
            kernel_size=2,
            padding=0,
            bias_attr=True,
            mesh=mesh1,
            stride=2,
            data_format="NHWC",
        )
        self._test_intermediate(
            N=4,
            C=6,
            H=28,
            W=28,
            kernel_size=3,
            padding=1,
            bias_attr=True,
            mesh=mesh2,
            data_format="NHWC",
        )


if __name__ == '__main__':
    tester = TestTPConv()
    tester.run_test_cases()
