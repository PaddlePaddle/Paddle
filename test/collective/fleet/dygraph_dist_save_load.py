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
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

import paddle
from paddle import distributed as dist
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel.sharding.group_sharded_optimizer_stage2 import (
    GroupShardedOptimizerStage2,
)
from paddle.distributed.fleet.meta_parallel.sharding.group_sharded_stage2 import (
    GroupShardedStage2,
)
from paddle.incubate.distributed.utils.io import load, save
from paddle.nn import Linear

print(load)
epoch = 2
linear_size = 1000


class MLP(paddle.nn.Layer):
    def __init__(self, linear_size=1000, param_attr=None, bias_attr=None):
        super().__init__()

        self._linear1 = Linear(linear_size, linear_size)
        self._linear2 = Linear(linear_size, linear_size)
        self._linear3 = Linear(linear_size, 10)

    def forward(self, inputs):
        y = self._linear1(inputs)
        y = self._linear2(y)
        y = self._linear3(y)
        return y


class RandomDataset(paddle.io.Dataset):
    def __init__(self, num_samples=2000, linear_size=1000):
        self.num_samples = num_samples
        self.linear_size = linear_size

    def __getitem__(self, idx):
        img = np.random.rand(self.linear_size).astype('float32')
        label = np.ones(1).astype('int64')
        return img, label

    def __len__(self):
        return self.num_samples


def optimizer_setting(model, use_pure_fp16, opt_group=False):
    clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0)
    optimizer = paddle.optimizer.AdamW(
        parameters=(
            [
                {
                    "params": model.parameters(),
                }
            ]
            if opt_group
            else model.parameters()
        ),
        learning_rate=0.001,
        weight_decay=0.00001,
        grad_clip=clip,
        multi_precision=use_pure_fp16,
    )

    return optimizer


def train_mlp(
    model,
    sharding_stage,
    batch_size=100,
    use_pure_fp16=False,
    accumulate_grad=False,
    opt_group=False,
    save_model=False,
    test_minimize=False,
    opt_state=None,
):
    if sharding_stage != "dp":
        group = paddle.distributed.new_group([0, 1], backend="nccl")
    if opt_group:
        optimizer = optimizer_setting(
            model=model, use_pure_fp16=use_pure_fp16, opt_group=opt_group
        )
    else:
        optimizer = optimizer_setting(model=model, use_pure_fp16=use_pure_fp16)

    if sharding_stage == 2:
        optimizer = GroupShardedOptimizerStage2(
            params=optimizer._parameter_list, optim=optimizer, group=group
        )
        model = GroupShardedStage2(
            model, optimizer, group=group, buffer_max_size=2**21
        )
        model._set_reduce_overlap(True)
        optimizer._set_broadcast_overlap(True, model)
    else:
        model = paddle.DataParallel(model)

    # check optimizer.minimize() error
    if test_minimize:
        try:
            optimizer.minimize()
        except:
            print(
                "====== Find sharding_stage2_optimizer.minimize() error ======"
            )
        return

    paddle.seed(2023)
    np.random.seed(2023)
    train_loader = paddle.io.DataLoader(
        RandomDataset(),
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=0,
    )

    if sharding_stage == 2:
        model.to(device="gpu")
    if opt_state is not None:
        optimizer.set_state_dict(opt_state)

    for eop in range(epoch):
        model.train()

        for batch_id, data in enumerate(train_loader()):
            img, label = data
            label.stop_gradient = True
            img.stop_gradient = True

            out = model(img)
            loss = paddle.nn.functional.cross_entropy(input=out, label=label)

            avg_loss = paddle.mean(x=loss.cast(dtype=paddle.float32))
            if batch_size == 20:
                avg_loss = avg_loss / 5
            avg_loss.backward()

            if not accumulate_grad:
                optimizer.step()
                optimizer.clear_grad()

        if accumulate_grad:
            optimizer.step()
            optimizer.clear_grad()

    paddle.device.cuda.synchronize()

    if save_model:
        return model, optimizer
    return model.parameters()


def save_model(model, output_dir, **configs):
    configs["save_model"] = True
    model, opt = train_mlp(model, **configs)

    model_file = os.path.join(
        output_dir, f"rank{dist.get_rank()}model.pdparams"
    )
    opt_file = os.path.join(output_dir, f"rank{dist.get_rank()}model.pdopt")

    g_model_file = os.path.join(
        output_dir, f"rank{dist.get_rank()}g_model.pdparams"
    )
    g_opt_file = os.path.join(output_dir, f"rank{dist.get_rank()}g_model.pdopt")

    paddle.save(model.state_dict(), model_file)
    paddle.save(opt.state_dict(), opt_file)

    save(
        model.state_dict(), g_model_file, gather_to=[0, 1], state_type="params"
    )
    save(opt.state_dict(), g_opt_file, gather_to=[0, 1], state_type="opt")


def load_mode(model, model_state_dict, output_param_path, **configs):
    configs["save_model"] = False
    model.set_state_dict(model_state_dict)
    params = train_mlp(model, **configs)
    paddle.save(params, output_param_path)


def step_check(path1, path2):
    m1 = paddle.load(path1)
    m2 = paddle.load(path2)
    for v1, v2 in zip(m1, m2):
        np.testing.assert_allclose(v1.numpy(), v2.numpy())
        print(f"value same: {v1.name}")


def step_save(strategy, output_dir, seed):
    python_exe = sys.executable
    # save data
    os.makedirs(output_dir + "/logs", exist_ok=True)
    filename = os.path.basename(__file__)
    cmd = (
        f"{python_exe} -m paddle.distributed.launch  --log_dir {output_dir}/logs"
        f" --gpus 0,1 {filename} --cmd save --strategy {strategy} --output_dir {output_dir} --seed {seed}"
    )
    p = subprocess.Popen(cmd.split())
    p.communicate()
    assert p.poll() == 0


def step_load(
    saved_strategy, current_strategy, saved_dir, load_way, output_path, seed
):
    python_exe = sys.executable
    os.makedirs(f"{saved_dir}/load/logs", exist_ok=True)
    filename = os.path.basename(__file__)
    # load dp
    cmd = (
        f"{python_exe} -m paddle.distributed.launch --log_dir {saved_dir}/load/logs"
        f" --gpus 0,1  {filename} --cmd load --strategy {current_strategy} --output_dir {saved_dir} --load_dir {saved_dir}/{saved_strategy}/save --load_way {load_way}"
        f" --output_param_path {output_path} --seed {seed}"
    )
    p = subprocess.Popen(cmd.split())
    p.communicate()
    assert p.poll() == 0


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
    elif args.strategy == "sharding_stage2":
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": 1,
            "pp_degree": 1,
            "sharding_degree": 2,
        }
    else:
        raise ValueError(f"Not supported strategy: {args.strategy}")

    fleet.init(is_collective=True, strategy=strategy)
    fleet.set_log_level("DEBUG")

    mlp1 = MLP()
    output_dir = os.path.join(args.output_dir, args.strategy, args.cmd)
    os.makedirs(output_dir, exist_ok=True)

    if args.cmd.lower() == "save":
        if args.strategy == "dp":
            # DP VS stage2
            save_model(
                mlp1,
                output_dir,
                sharding_stage="dp",
                use_pure_fp16=False,
                opt_group=False,
                save_model=True,
            )
        elif args.strategy == "sharding_stage2":
            save_model(
                mlp1,
                output_dir,
                sharding_stage=2,
                use_pure_fp16=False,
                opt_group=False,
                save_model=True,
            )
        else:
            raise ValueError(f"Not supported {args.strategy}")
    elif args.cmd.lower() == "load":
        output_dir = args.load_dir
        model_file = os.path.join(
            output_dir, f"rank{dist.get_rank()}model.pdparams"
        )
        opt_file = os.path.join(output_dir, f"rank{dist.get_rank()}model.pdopt")
        g_model_file = os.path.join(
            output_dir, f"rank{args.gather_to}g_model.pdparams"
        )
        g_opt_file = os.path.join(
            output_dir, f"rank{args.gather_to}g_model.pdopt"
        )

        if args.load_way == "load":
            model_file = g_model_file
            opt_file = g_opt_file
            load_ = lambda x: eval(args.load_way)(x, place='cpu')
        else:
            load_ = eval(args.load_way)

        model = load_(model_file)
        opt = load_(opt_file)
        for k in opt.keys():
            print("opt k:", k)
        if args.strategy == "dp":
            load_mode(
                mlp1,
                model,
                args.output_param_path,
                sharding_stage="dp",
                use_pure_fp16=False,
                opt_group=False,
                save_model=False,
                opt_state=opt,
            )
        elif args.strategy == "sharding_stage2":
            load_mode(
                mlp1,
                model,
                args.output_param_path,
                sharding_stage=2,
                use_pure_fp16=False,
                opt_group=False,
                save_model=False,
                opt_state=opt,
            )
        else:
            raise ValueError(f"Not supported strategy {args.strategy}")

    else:
        raise ValueError(f"Not supported cmd: {args.cmd}")


def run_case(args):
    saving_strategy = args.test_case.split(":")[0]
    loading_strategy = args.test_case.split(":")[1]

    output_dir = tempfile.mkdtemp()
    print("output dir:", output_dir)
    os.makedirs(output_dir + "/load_save", exist_ok=True)
    # save dp
    step_save(saving_strategy, output_dir, args.seed)
    # return

    # load dp
    p1 = os.path.join(output_dir, "m1.pdparams")
    p2 = os.path.join(output_dir, "m2.pdparams")

    step_load(
        saving_strategy,
        saving_strategy,
        output_dir,
        "paddle.load",
        p1,
        args.seed + 1,
    )
    step_load(
        saving_strategy, loading_strategy, output_dir, "load", p2, args.seed + 2
    )

    # check
    step_check(p1, p2)

    shutil.rmtree(output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cmd", default="main", choices=["main", "save", "load"]
    )
    parser.add_argument(
        "--strategy", required=False, choices=["dp", "sharding_stage2"]
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
            "dp:dp",
            "dp:sharding_stage2",
            "sharding_stage2:dp",
            "sharding_stage2:sharding_stage2",
        ],
    )
    parser.add_argument("--gather_to", required=False, default=0)
    parser.add_argument("--seed", type=int, default=2022)

    args = parser.parse_args()
    test_save_load(args)
