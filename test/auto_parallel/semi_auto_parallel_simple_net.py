# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import random

import numpy as np

import paddle
import paddle.distributed as dist
from paddle import nn
from paddle.distributed.fleet.utils import recompute

BATCH_SIZE = 16
BATCH_NUM = 4
IMAGE_SIZE = 784
CLASS_NUM = 10


def create_numpy_like_random(name):
    return paddle.ParamAttr(
        name=name, initializer=paddle.nn.initializer.Uniform(0, 1)
    )


class DemoNet(nn.Layer):
    def __init__(
        self,
        param_prefix="",
        is_recompute=False,
        is_pp=False,
        pp_reshard_dist_attr=None,
    ):
        super().__init__()
        weight_attr_0 = create_numpy_like_random(param_prefix + "_0")
        weight_attr_1 = create_numpy_like_random(param_prefix + "_1")

        self.is_pp = is_pp
        self.is_recompute = is_recompute
        self.pp_reshard_dist_attr = pp_reshard_dist_attr
        self.linear_0 = nn.Linear(IMAGE_SIZE, IMAGE_SIZE, weight_attr_0)
        self.linear_1 = nn.Linear(IMAGE_SIZE, CLASS_NUM, weight_attr_1)
        self.relu = nn.ReLU()

    def _inner_forward_fn(self, x):
        out = self.linear_0(x)
        out = self.relu(out)
        if self.is_pp:
            out = dist.reshard(out, self.pp_reshard_dist_attr)
        out = self.linear_1(out)
        return out

    def forward(self, x):
        if self.is_recompute:
            return recompute(self._inner_forward_fn, x)
        else:
            return self._inner_forward_fn(x)


class TestSimpleNetForSemiAutoParallel:
    def __init__(self):
        self._dtype = os.getenv("dtype")
        self._backend = os.getenv("backend")
        self._seed = eval(os.getenv("seed"))
        self._mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
        self._pp_mesh0 = dist.ProcessMesh([0], dim_names=["x"])
        self._pp_mesh1 = dist.ProcessMesh([1], dim_names=["x"])
        self.pp_reshard_dist_attr = dist.DistAttr(
            mesh=self._pp_mesh1, sharding_specs=[None, None]
        )

        paddle.set_device(self._backend)
        self.init_single_card_net_result()

    def shard_fn(self, layer_name, layer, process_mesh):
        if layer_name == 'linear_0':
            dist_attr = dist.DistAttr(
                mesh=process_mesh, sharding_specs=[None, 'x']
            )
            layer.weight = dist.shard_tensor(layer.weight, dist_attr=dist_attr)
        elif layer_name == 'linear_1':
            dist_attr = dist.DistAttr(
                mesh=process_mesh, sharding_specs=['x', None]
            )
            layer.weight = dist.shard_tensor(layer.weight, dist_attr=dist_attr)

    def pp_shard_fn(self, layer_name, layer, process_mesh):
        if layer_name == 'linear_0':
            # shard_layer doens't support cross-mesh now.
            # input process_mesh of pp_shard_fn is useless,
            # it's defined just for unified format.
            weight_dist_attr = dist.DistAttr(
                mesh=self._pp_mesh0, sharding_specs=[None, None]
            )
            bias_dist_attr = dist.DistAttr(
                mesh=self._pp_mesh0, sharding_specs=[None]
            )
            layer.weight = dist.shard_tensor(
                layer.weight, dist_attr=weight_dist_attr
            )
            layer.bias = dist.shard_tensor(layer.bias, dist_attr=bias_dist_attr)
        elif layer_name == 'linear_1':
            weight_dist_attr = dist.DistAttr(
                mesh=self._pp_mesh1, sharding_specs=[None, None]
            )
            bias_dist_attr = dist.DistAttr(
                mesh=self._pp_mesh1, sharding_specs=[None]
            )
            layer.weight = dist.shard_tensor(
                layer.weight, dist_attr=weight_dist_attr
            )
            layer.bias = dist.shard_tensor(layer.bias, dist_attr=bias_dist_attr)

    def set_random_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        paddle.seed(seed)

    def init_input_data(self):
        image = np.random.random([BATCH_SIZE, IMAGE_SIZE]).astype('float32')
        label = np.random.random([BATCH_SIZE, CLASS_NUM]).astype('float32')
        return paddle.to_tensor(image), paddle.to_tensor(label)

    def run_dynamic(self, layer, shard_input=False, is_pp=False):
        # create loss
        loss_fn = nn.MSELoss()
        # run forward and backward
        input_mesh = self._pp_mesh0 if is_pp else self._mesh
        opt = paddle.optimizer.SGD(
            learning_rate=0.1, parameters=layer.parameters()
        )
        # TODO: solve the derivation issue of AdamW
        # for _ in range(5):
        for _ in range(1):
            image, label = self.init_input_data()
            if shard_input:
                image = dist.shard_tensor(
                    image,
                    dist_attr=dist.DistAttr(
                        mesh=input_mesh, sharding_specs=['x', None]
                    ),
                )

            out = layer(image)
            loss = loss_fn(out, label)
            loss.backward()

            opt.step()
            opt.clear_grad()
        return loss, layer.parameters()

    def init_single_card_net_result(self):
        self.set_random_seed(self._seed)
        self.base_loss, self.base_parameters = self.run_dynamic(
            DemoNet("demo_weight")
        )

    def check_tensor_eq(self, a, b, rtol=1e-05, atol=0, verbose=True):
        np1 = a.astype("float32").numpy()
        np2 = b.astype("float32").numpy()
        np.testing.assert_allclose(
            np1, np2, rtol=rtol, atol=atol, verbose=verbose
        )

    def test_dp_demo_net(self):
        self.set_random_seed(self._seed)

        self.dp_loss, self.dp_parameters = self.run_dynamic(
            DemoNet("dp_demo_weight"),
            shard_input=True,
        )
        self.check_tensor_eq(self.dp_loss, self.base_loss)
        for param, param_base in zip(self.dp_parameters, self.base_parameters):
            self.check_tensor_eq(param, param_base)
            self.check_tensor_eq(param.grad, param_base.grad)

    def test_mp_demo_net(self):
        self.set_random_seed(self._seed)

        mp_layer = dist.shard_layer(
            DemoNet("mp_demo_weight"), self._mesh, self.shard_fn
        )

        self.mp_loss, self.mp_parameters = self.run_dynamic(mp_layer)
        self.check_tensor_eq(self.mp_loss, self.base_loss)

        for param, param_base in zip(self.mp_parameters, self.base_parameters):
            self.check_tensor_eq(param, param_base)
            self.check_tensor_eq(param.grad, param_base.grad)

    def test_pp_demo_net(self):
        self.set_random_seed(self._seed)

        # Send/Recv operators doens't support CPU now.
        if self._backend != "gpu":
            return

        pp_layer = dist.shard_layer(
            DemoNet(
                "pp_demo_weight",
                is_pp=True,
                pp_reshard_dist_attr=self.pp_reshard_dist_attr,
            ),
            self._pp_mesh0,
            self.pp_shard_fn,
        )

        self.pp_loss, self.pp_parameters = self.run_dynamic(
            pp_layer, is_pp=True
        )

        rank = dist.get_rank()
        # TODO(GhostScreaming): DistTensor.numpy() doesn't support
        # cross-mesh now, ReshardXToReplicated function in eager_method
        # needs to be fixed later.
        if rank == 0:
            # linear_0 weight and bias
            self.check_tensor_eq(self.pp_parameters[0], self.base_parameters[0])
            self.check_tensor_eq(self.pp_parameters[1], self.base_parameters[1])
        else:
            self.check_tensor_eq(self.pp_loss, self.base_loss)
            # linear_1 weight and bias
            self.check_tensor_eq(self.pp_parameters[2], self.base_parameters[2])
            self.check_tensor_eq(self.pp_parameters[3], self.base_parameters[3])

        # TODO(GhostScreaming): Enable it later.
        # for param, param_base in zip(self.mp_parameters, self.base_parameters):
        #     self.check_tensor_eq(param, param_base)
        #     self.check_tensor_eq(param.grad, param_base.grad)

    def run_test_case(self):
        self.test_dp_demo_net()
        self.test_mp_demo_net()
        self.test_pp_demo_net()


if __name__ == '__main__':
    TestSimpleNetForSemiAutoParallel().run_test_case()
