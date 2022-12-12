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

import argparse
import copy
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

import paddle
import paddle.fluid as fluid
from paddle import distributed as dist
from paddle.distributed import fleet
from paddle.distributed.auto_parallel import engine
from paddle.distributed.fleet.layers.mpu.mp_layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from paddle.distributed.fleet.meta_parallel.parallel_layers.pp_layers import (
    LayerDesc,
    PipelineLayer,
)
from paddle.distributed.sharding.group_sharded import group_sharded_parallel
from paddle.distributed.utils.log_utils import get_logger
from paddle.fluid.dataloader.dataset import IterableDataset
from paddle.incubate.distributed.utils.io import save_for_auto_inference
from paddle.nn import Linear

logger = get_logger("INFO", __file__)


epoch = 2
linear_size = 1000


class MLP_pipe(PipelineLayer):
    def __init__(
        self,
        embedding_size=1000,
        linear_size=1000,
        param_attr=None,
        bias_attr=None,
    ):
        desc = [
            LayerDesc(
                VocabParallelEmbedding,
                num_embeddings=embedding_size,
                embedding_dim=linear_size,
            ),
            LayerDesc(
                RowParallelLinear,
                in_features=linear_size,
                out_features=linear_size,
                has_bias=True,
            ),
            LayerDesc(
                ColumnParallelLinear,
                in_features=linear_size,
                out_features=linear_size,
                gather_output=True,
                has_bias=True,
            ),
            LayerDesc(Linear, in_features=linear_size, out_features=10),
        ]
        super(MLP_pipe, self).__init__(
            desc,
            num_stages=2,
            loss_fn=paddle.nn.CrossEntropyLoss(),
            topology=fleet.get_hybrid_communicate_group()._topo,
        )


class MLP_Hybrid(fluid.Layer):
    def __init__(
        self,
        embedding_size=1000,
        linear_size=1000,
        param_attr=None,
        bias_attr=None,
    ):
        super(MLP_Hybrid, self).__init__()
        self.embedding = VocabParallelEmbedding(embedding_size, linear_size)
        self._linear1 = RowParallelLinear(
            linear_size, linear_size, has_bias=True, input_is_parallel=True
        )
        self._linear2 = ColumnParallelLinear(
            linear_size, linear_size, gather_output=True, has_bias=True
        )
        self._linear3 = Linear(linear_size, 10)

    def forward(self, src):
        inputs = self.embedding(src)
        # slice for a bug in row parallel linear
        mp_group = (
            fleet.get_hybrid_communicate_group().get_model_parallel_group()
        )
        step = inputs.shape[-1] // mp_group.nranks
        mp_rank = dist.get_rank(mp_group)
        mp_rank = mp_rank if mp_rank >= 0 else 0
        inputs = inputs[..., step * mp_rank : step * mp_rank + step]

        y = self._linear1(inputs)
        y = self._linear2(y)
        y = self._linear3(y)
        return y


class MLP(fluid.Layer):
    def __init__(
        self,
        embedding_size=1000,
        linear_size=1000,
        param_attr=None,
        bias_attr=None,
    ):
        super(MLP, self).__init__()
        self.embedding = paddle.nn.Embedding(embedding_size, linear_size)
        self._linear1 = Linear(linear_size, linear_size)
        self._linear2 = Linear(linear_size, linear_size)
        self._linear3 = Linear(linear_size, 10)

    def forward(self, src):
        inputs = self.embedding(src)
        y = self._linear1(inputs)
        y = self._linear2(y)
        y = self._linear3(y)
        return y


def gen_uniq_random_numbers(low, high, size, seed):
    assert np.prod(size) <= high - low
    pool = list(range(low, high))
    data = np.zeros(size).astype("int32").reshape(-1)
    np.random.seed(10245)
    for i in range(np.prod(size)):
        pos = int(np.random.randint(0, len(pool)))
        data[i] = pool[pos]
        pool.remove(pool[pos])
    np.random.seed(seed)
    return data.reshape(size)


class RangeIterableDataset(IterableDataset):
    def __init__(
        self, data_path, ebd=1000, start=0, end=100, linear_size=1000, seed=1024
    ):
        self.start = start
        self.end = end
        self.img = gen_uniq_random_numbers(0, 1000, (100, 1), seed)

    def __iter__(self):
        for idx in range(self.start, self.end):
            label = np.ones(1).astype('int32')
            yield self.img[idx], label


def optimizer_setting(args, model):
    optimizer = paddle.optimizer.SGD(
        learning_rate=0.0 if args.strategy == "static" else 0.01,
        parameters=model.parameters(),
        weight_decay=0.01,
    )

    return optimizer


def train_mlp(args, model, loss, opt_state=None, save_model=False):
    optimizer = optimizer_setting(args, model=model)

    if args.strategy in ["mp", "dp", "pp"]:
        model = fleet.distributed_model(model)
        optimizer = fleet.distributed_optimizer(optimizer)
    elif args.strategy == "sharding_stage2":
        model, optimizer, _ = wrap_sharding_2_3(
            model, optimizer, None, False, 2
        )
    elif args.strategy == "sharding_stage3":
        model, optimizer, _ = wrap_sharding_2_3(
            model, optimizer, None, False, 3
        )
    elif args.strategy != "single":
        raise ValueError(f"not supported strategy: {args.strategy}")

    dataset = RangeIterableDataset(
        data_path=os.path.join(args.output_dir, "data.npy"), seed=args.seed
    )

    train_loader = paddle.io.DataLoader(dataset, batch_size=100, drop_last=True)

    if dist.get_world_size() > 1:
        pp_degree = (
            fleet.get_hybrid_communicate_group().get_pipe_parallel_world_size()
        )
    else:
        pp_degree = 0

    model.train()
    for epo in range(epoch):
        for step, data in enumerate(train_loader()):
            img, label = data
            label.stop_gradient = True
            img.stop_gradient = True
            if pp_degree <= 1:
                out = model(img)
                avg_loss = loss(out, label)
                paddle.device.cuda.synchronize()
                avg_loss.backward()
                optimizer.step()
            else:
                avg_loss = model.train_batch(data, optimizer)

    model.eval()
    print("=============== predict in dygraph mode =================")
    for step, data in enumerate(train_loader()):
        img, label = data
        if pp_degree <= 1:
            out = model(img)
            out = out.numpy()
        else:
            out = model.eval_batch(data)
            out = np.array(out)

    paddle.device.cuda.synchronize()
    if save_model:
        return model, optimizer, out

    return None


def train_mlp_static(args, model, loss, opt_state=None, save_model=False):
    optimizer = optimizer_setting(args, model=model)
    model = engine.Engine(model, loss=loss, optimizer=optimizer, strategy=None)

    dataset = RangeIterableDataset(
        data_path=os.path.join(args.output_dir, "data.npy"), seed=args.seed
    )
    model.load(os.path.join(args.load_dir, "saved"), load_optimizer=False)
    model.fit(dataset, epochs=1)
    model.save(os.path.join(args.output_dir, "static_save"))
    paddle.device.cuda.synchronize()
    print("=============== predict in static mode =================")
    out = model.predict(dataset, verbose=1000)

    if save_model:
        return model, optimizer
    return out


def step_check(output_dir):
    p1 = os.path.join(output_dir, "static.npy")
    p2 = os.path.join(output_dir, "dygraph.npy")
    m1 = np.load(p1).reshape(-1)
    m2 = np.load(p2).reshape(-1)
    try:
        assert np.allclose(m1, m2, rtol=1e-5, atol=1e-6)
    except:
        diff = m1 - m2
        logger.error(f"max diff{diff.max()}, min diff: {diff.min()}")
        logger.error(f"{m1[:10]}")
        logger.error(f"{m2[:10]}")
        raise ValueError("diff is too large")


def step_save(strategy, output_dir, seed):
    python_exe = sys.executable
    # save data
    os.makedirs(output_dir + "/logs", exist_ok=True)
    filename = os.path.basename(__file__)
    if strategy != "single":
        cmd = (
            f"{python_exe} -m paddle.distributed.launch  --log_dir {output_dir}/logs"
            f" --gpus 0,1 {filename} --cmd save --strategy {strategy} --output_dir {output_dir} --seed {seed}"
        )
    else:
        cmd = f"{python_exe} {filename} --cmd save --strategy {strategy} --output_dir {output_dir} --seed {seed}"

    logger.info(f"exe: {cmd}")
    p = subprocess.Popen(cmd.split())
    p.communicate()
    assert p.poll() == 0


def step_load(curent_strateggy, saved_dir, seed):
    python_exe = sys.executable
    os.makedirs(f"{saved_dir}/load/logs", exist_ok=True)
    filename = os.path.basename(__file__)
    # load dp
    cmd = (
        f"{python_exe} -m paddle.distributed.launch --log_dir {saved_dir}/load/logs"
        f" --gpus 0  {filename} --cmd load --strategy {curent_strateggy} --output_dir {saved_dir} --load_dir {saved_dir} --seed {seed}"
    )
    logger.info(f"exe: {cmd}")
    env = copy.copy(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    p = subprocess.Popen(cmd.split(), env=env)
    p.communicate()
    assert p.poll() == 0


def wrap_sharding_2_3(model, optimizer, scaler, sharding_offload, stage):
    group = fleet.get_hybrid_communicate_group().get_sharding_parallel_group()
    level = "p_g_os" if stage == 3 else "os_g"
    return group_sharded_parallel(
        model=model,
        optimizer=optimizer,
        level=level,
        scaler=scaler,
        group=group,
        offload=sharding_offload,
    )


def test_save_load(args):

    np.random.seed(args.seed)
    paddle.seed(args.seed)

    if args.cmd == "main":
        run_case(args)
        return

    paddle.distributed.init_parallel_env()
    strategy = fleet.DistributedStrategy()
    if args.strategy == "dp":
        strategy.hybrid_configs = {
            "dp_degree": 2,
            "mp_degree": 1,
            "pp_degree": 1,
            "sharding_degree": 1,
        }
    elif args.strategy in ["sharding_stage2", "sharding_stage3"]:
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": 1,
            "pp_degree": 1,
            "sharding_degree": 2,
        }
    elif args.strategy == "mp":
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": 2,
            "pp_degree": 1,
            "sharding_degree": 1,
        }
    elif args.strategy == "pp":
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": 1,
            "pp_degree": 2,
            "sharding_degree": 1,
        }
        strategy.pipeline_configs = {
            "accumulate_steps": 10,
            "micro_batch_size": 10,
        }
    elif args.strategy == "static":
        paddle.enable_static()
    elif args.strategy != "single":
        raise ValueError(f"Not supported strategy: {args.strategy}")

    loss = paddle.nn.CrossEntropyLoss()

    fleet.set_log_level("INFO")
    if dist.get_world_size() <= 1:
        mlp1 = MLP()
        if args.strategy == "static":
            out_static = train_mlp_static(args, mlp1, loss, save_model=False)
            np.save(os.path.join(args.output_dir, "static.npy"), out_static)
        else:
            model, _, out_dygraph = train_mlp(args, mlp1, loss, save_model=True)
            np.save(os.path.join(args.output_dir, "dygraph.npy"), out_dygraph)
    else:
        fleet.init(is_collective=True, strategy=strategy)
        pp_group = (
            fleet.get_hybrid_communicate_group().get_pipe_parallel_group()
        )
        if pp_group.nranks > 1:
            mlp1 = MLP_pipe()
        else:
            mlp1 = MLP_Hybrid()
        model, _, out_dygraph = train_mlp(args, mlp1, loss, save_model=True)
        if (
            dist.get_world_size() == 0
            or dist.get_rank() == dist.get_world_size() - 1
        ):
            np.save(os.path.join(args.output_dir, "dygraph.npy"), out_dygraph)

    if args.cmd == "save":
        save_for_auto_inference(os.path.join(args.output_dir, "saved"), model)


def run_case(args):

    saving_strategy = args.test_case.split(":")[0]
    loading_strategy = args.test_case.split(":")[1]

    output_dir = tempfile.mkdtemp()
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    try:
        step_save(saving_strategy, output_dir, args.seed)
        step_load(loading_strategy, output_dir, args.seed + 1)
        step_check(output_dir)
    except Exception as e:
        shutil.rmtree(output_dir)
        raise RuntimeError(f"Test failed.\n {e.__str__()}")
    shutil.rmtree(output_dir)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cmd", default="main", choices=["main", "save", "load"]
    )
    parser.add_argument(
        "--strategy",
        required=False,
        choices=[
            "single",
            "dp",
            "mp",
            "pp",
            "sharding_stage2",
            "sharding_stage3",
            "static",
        ],
    )
    parser.add_argument(
        "--load_way", choices=["paddle.load", "load"], required=False
    )
    parser.add_argument("--load_dir", required=False)
    parser.add_argument("--output_dir", required=False)
    parser.add_argument("--output_param_path", required=False)
    parser.add_argument(
        "--test_case",
        required=False,
        choices=[
            "dp:static",
            "mp:static",
            "pp:static",
            "sharding_stage2:static",
            "sharding_stage3:static",
            "single:static",
        ],
    )
    parser.add_argument("--gather_to", required=False, default=0)
    parser.add_argument("--seed", type=int, default=2022)

    args = parser.parse_args()
    test_save_load(args)
