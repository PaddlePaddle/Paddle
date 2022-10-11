# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import paddle
import paddle.distributed.fleet as fleet
from test_collective_multi_nodes import TestCollectiveAPIRunnerBase, runtime_main
from paddle import nn
import numpy as np

from paddle.distributed.fleet.utils import recompute


def weight_init(mp, shape, col=True, seed=1024):
    np.random.seed(seed)
    w = np.random.normal(0, 0.02, size=shape)
    if mp is None:
        _w = w
    else:
        if col:
            step = shape[1] // mp.nranks
            _w = w[:, mp.rank * step:mp.rank * step + step]
        else:
            step = shape[0] // mp.nranks
            _w = w[mp.rank * step:mp.rank * step + step, :]
    return paddle.fluid.initializer.NumpyArrayInitializer(_w)


class Criterion(nn.Layer):

    def __init__(self):
        super(Criterion, self).__init__()
        self.loss_func = nn.MSELoss(reduction="mean")

    def forward(self, pred, label):
        loss = self.loss_func(pred, label)
        return loss


class RecomputeMatmulBlock(nn.Layer):

    def __init__(self, mp, seed, m, n, k):
        super(RecomputeMatmulBlock, self).__init__()
        self.mp = mp
        if mp is not None and mp.nranks > 1:
            mp_linear_1 = fleet.meta_parallel.ColumnParallelLinear(
                m,
                n,
                weight_attr=weight_init(mp, (m, n), True, seed),
                has_bias=True,
                gather_output=False)
            mp_linear_2 = fleet.meta_parallel.RowParallelLinear(
                n,
                k,
                weight_attr=weight_init(mp, (n, k), False, seed + 1),
                has_bias=True,
                input_is_parallel=True)
        else:
            mp_linear_1 = nn.Linear(m,
                                    n,
                                    weight_attr=weight_init(
                                        None, (m, n), True, seed))
            mp_linear_2 = nn.Linear(n,
                                    k,
                                    weight_attr=weight_init(
                                        None, (n, k), True, seed + 1))
        self.layers = nn.Sequential(mp_linear_1, mp_linear_2)

    def forward(self, x):
        if self.mp:
            return recompute(self.layers, x)
        else:
            return self.layers(x)


RecomputeBlock = RecomputeMatmulBlock


class ModelPipeline(fleet.meta_parallel.PipelineLayer):

    def __init__(self, hcg):
        paddle.seed(1024)
        dp_linear = nn.Linear(32, 64)
        self.layers_pp = []
        self.topology = hcg.topology()
        self.layers_pp.append(dp_linear)
        mp = hcg.get_model_parallel_group()
        for i in range(6):
            mp_layer = RecomputeBlock(mp, 1024 + i, 64, 128, 64)
            act = nn.ReLU6()
            layer_seq = nn.Sequential(mp_layer, act)
            self.layers_pp.append(layer_seq)

        out = nn.Linear(64, 32)
        self.layers_pp.append(out)
        super(ModelPipeline, self).__init__(layers=self.layers_pp,
                                            loss_fn=Criterion(),
                                            topology=self.topology)


class Model(nn.Layer):

    def __init__(self, hcg):
        super(Model, self).__init__()
        paddle.seed(1024)
        dp_linear = nn.Linear(32, 64)
        self.layers_pp = []
        self.layers_pp.append(dp_linear)
        mp = hcg.get_model_parallel_group() if hcg else None
        for i in range(6):
            mp_layer = RecomputeBlock(mp, 1024 + i, 64, 128, 64)
            act = nn.ReLU6()
            layer_seq = nn.Sequential(mp_layer, act)
            self.layers_pp.append(layer_seq)

        out = nn.Linear(64, 32)
        self.layers_pp.append(out)
        self.layers = nn.Sequential(*self.layers_pp)

    def forward(self, x):
        return self.layers(x)


class TestDygrapgHybridRecompute(TestCollectiveAPIRunnerBase):

    def __init__(self):
        pass

    def check_pass(self, *args, **kwargs):

        from common import init_parallel_env
        import paddle
        from paddle.distributed import fleet
        hcg = init_parallel_env("DP4-MP2-PP2-SH1-O1", 64)
        pp_degree = hcg.get_pipe_parallel_world_size()
        import numpy as np
        crit = Criterion()
        if pp_degree <= 1:
            model = Model(hcg)
        else:
            model = ModelPipeline(hcg)

        model_base = Model(None)

        optimizer = paddle.optimizer.Adam(learning_rate=0.01,
                                          parameters=model.parameters())
        optimizer_base = paddle.optimizer.Adam(
            learning_rate=0.01, parameters=model_base.parameters())

        model = fleet.distributed_model(model)
        optimizer = fleet.distributed_optimizer(optimizer)
        loss_hybrid_arr = []
        loss_base_arr = []

        x = paddle.to_tensor(np.random.random((16, 32))).astype("float32")
        y = paddle.to_tensor(np.random.random((16, 32))).astype("float32")

        for _ in range(5):
            if pp_degree > 1:
                loss = model.train_batch([x, y], optimizer=optimizer)
            else:
                output = model(x)
                loss = crit(output, y)
                loss.backward()
                optimizer.step()
                optimizer.clear_grad()

            # baseline loss
            output_base = model_base(x)
            loss_base = crit(output_base, y)
            loss_base.backward()
            optimizer_base.step()
            optimizer_base.clear_grad()

            loss_base_arr.append(loss_base.numpy())
            loss_hybrid_arr.append(loss)
        assert np.allclose(loss_base_arr, loss_hybrid_arr, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    runtime_main(TestDygrapgHybridRecompute, "dpppmp")
