# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import atexit
import copy
import multiprocessing
import os
import random
import sys
import unittest
from functools import wraps
from multiprocessing.reduction import DupFd

import numpy as np

# 2) Hook pybind: DenseTensor._share_cuda / _new_shared_cuda
try:
    import paddle
    from paddle.base import core

    _orig_share_vmm = core.DenseTensor._share_vmm
    _orig_new_shared_vmm = core.DenseTensor._new_shared_vmm

    @wraps(_orig_share_vmm)
    def _spy_share_vmm(self):
        """
        发送端：导出 VMM 元信息，并且把 fd 用 DupFd 包起来，确保经由 mp 管道以 SCM_RIGHTS 传递。
        """
        meta = _orig_share_vmm(self)
        try:
            fd, off, size, dtype, dims, lod, dev = meta
            fd_safely = fd
            fd_safely = DupFd(fd)  # ★★ 核心：包装 fd ★★
            meta = (fd_safely, off, size, dtype, dims, lod, dev)
            print(
                f"[IPC-TX] pid={os.getpid()} "
                f"_share_vmm dev={dev} fd(type)={type(fd_safely).__name__} off={off} size={size}B",
                flush=True,
            )
            print(
                f"[TX] before put: type(fd)={type(fd).__name__}  off={off}  size={size}  dev={dev}",
                flush=True,
            )
            # traceback.print_stack(limit=10, file=sys.stdout)
        except Exception as e:
            print(
                f"[IPC-TX] _share_vmm meta decode failed: {e}",
                file=sys.stderr,
                flush=True,
            )
        return meta

    @wraps(_orig_new_shared_vmm)
    def _spy_new_shared_vmm(meta):
        """
        接收端：把 DupFd 转为当前进程有效的 int，再交给 C++ 导入。
        """
        try:
            fd, off, size, dtype, dims, lod, dev = meta
            print(
                f"[RX] before _new_shared_vmm: type(fd)={type(fd).__name__}  repr(fd)={fd}",
                flush=True,
            )
            print("before set device: ", core.get_cuda_current_device_id())
            core.set_cuda_current_device_id(dev)
            print("before set device: ", core.get_cuda_current_device_id())
            if hasattr(fd, "detach"):
                fd = fd.detach()  # ★★ 核心：detach 成 int ★★
                print(f"[RX] after detach: fd(int)={fd}", flush=True)
            # 立刻做一次探测，提前暴露 EBADF（定位更快）
            import os as _os

            _os.fstat(fd)
            print(
                f"[IPC-RX] pid={os.getpid()} "
                f"_new_shared_vmm dev={dev} fd(int)={fd} off={off} size={size}B",
                flush=True,
            )
            # traceback.print_stack(limit=10, file=sys.stdout)
            meta = (
                fd,
                off,
                size,
                dtype,
                dims,
                lod,
                dev,
            )  # 用 detch 后的 fd 覆盖
        except Exception as e:
            print(
                f"[IPC-RX] _new_shared_vmm precheck failed: {e}",
                file=sys.stderr,
                flush=True,
            )
        return _orig_new_shared_vmm(meta)

    # ★★ 正确绑定（你之前错绑到了 _share_cuda/_new_shared_cuda）★★
    core.DenseTensor._share_vmm = _spy_share_vmm
    core.DenseTensor._new_shared_vmm = _spy_new_shared_vmm
except Exception as e:
    print(
        f"[IPC-HOOK] core hook failed: {e}",
        file=sys.stderr,
        flush=True,
    )

import sys
from functools import wraps

import paddle
import paddle.distributed as dist
from paddle.base import core
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer.dygraph_sharding_optimizer import (
    DygraphShardingOptimizer,
    DygraphShardingOptimizerV2,
)
from paddle.distributed.fleet.utils.mix_precision_utils import (
    MixPrecisionLayer,
    MixPrecisionOptimizer,
)
from paddle.optimizer.fusion_utils import FusionStorageHelper

g_shard_split_param = int(os.environ.get("FLAGS_shard_split_param", 0))
g_shard_param_with_color = int(
    os.environ.get("FLAGS_shard_param_with_color", 0)
)

vocab_size = 20
hidden_size = 10
inner_size = 8
output_size = 10
seq_length = 2
batch_size = 4
STEPS = 10

DO_FUSE_OPTIMIZER = 0
DO_SYNC_PARAM = 1
DO_RETURN_DICT = 2


def parallel_matmul(lm_output, logit_weights, parallel_output):
    hcg = fleet.get_hybrid_communicate_group()
    model_parallel_group = hcg.get_model_parallel_group()
    world_size = hcg.get_model_parallel_world_size()
    rank = hcg.get_model_parallel_rank()

    if world_size > 1:
        input_parallel = paddle.distributed.collective._c_identity(
            lm_output, group=model_parallel_group
        )

        logits = paddle.matmul(input_parallel, logit_weights, transpose_y=True)

        if parallel_output:
            return logits

        return paddle.distributed.collective._c_concat(
            logits, group=model_parallel_group
        )
    else:
        logits = paddle.matmul(lm_output, logit_weights, transpose_y=True)
        return logits


class SimpleDPNet(paddle.nn.Layer):
    def __init__(
        self, vocab_size, hidden_size, inner_size, output_size, np_fc1, np_fc2
    ):
        super().__init__()
        self.linear1 = paddle.nn.Linear(
            hidden_size,
            inner_size,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Assign(np_fc1)
            ),
            bias_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Constant(0.0)
            ),
        )

        self.linear2 = paddle.nn.Linear(
            inner_size,
            hidden_size,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Assign(np_fc2)
            ),
            bias_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Constant(0.0)
            ),
        )

        self.linear3 = paddle.nn.Linear(
            hidden_size,
            output_size,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Constant(0.0)
            ),
            bias_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Constant(0.0)
            ),
        )

        self.embedding = paddle.nn.Embedding(
            vocab_size,
            hidden_size,
            weight_attr=paddle.nn.initializer.Constant(value=0.5),
        )

        if g_shard_param_with_color:
            for p in self.linear1.parameters():
                p.color = "linear1"

            for p in self.linear2.parameters():
                p.color = "linear2"

            for p in self.linear3.parameters():
                p.color = "linear3"

    def forward(self, x):
        x = self.embedding(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = paddle.matmul(x, self.embedding.weight, transpose_y=True)
        return x


class FusionWorker(multiprocessing.Process):
    def __init__(self, worker_id, device_id, task_queue, result_queue):
        super().__init__()
        self.worker_id = worker_id
        self.device_id = device_id
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.fusion_storage_helper = None

    def run(self):
        core.set_cuda_current_device_id(self.device_id)
        paddle.set_device(f"gpu:{self.device_id}")
        while True:
            task = self.task_queue.get()
            if task is None:
                self.task_queue.put(None)
                self.result_queue.put((self.worker_id, None))
                break

            task_type, task_body = task
            if task_type == DO_FUSE_OPTIMIZER:
                print("====== debug DO_FUSE_OPTIMIZER")
                self.build_fusion_storage_helper(task_body)
            elif task_type == DO_SYNC_PARAM:
                print("====== debug DO_SYNC_PARAM")
                self.fusion_storage_helper.sync_param()
                self.fusion_storage_helper.wait_all()
            elif task_type == DO_RETURN_DICT:
                print("====== debug DO_RETURN_DICT")
                result = self.fusion_storage_helper.state_dict()
                self.result_queue.put((self.worker_id, result))
            else:
                raise ValueError(f"Unknown task type: {task_type}")

    def build_fusion_storage_helper(self, task_body):
        (
            accumulators_meta,
            master_weights_meta,
            merged_model_params_meta,
            buffer_ipc_meta,
        ) = task_body
        if self.fusion_storage_helper is None:
            print("============== fusion_storage_helper is none  ==========")
            self.fusion_storage_helper = FusionStorageHelper(
                accumulators_meta,
                master_weights_meta,
                merged_model_params_meta,
                buffer_ipc_meta,
            )
        else:
            print("============== reset_meta ==========")
            self.fusion_storage_helper.reset_meta(
                accumulators_meta,
                master_weights_meta,
                merged_model_params_meta,
                buffer_ipc_meta,
            )


class TestDistMPTraining(unittest.TestCase):
    def setUp(self):
        random.seed(2021)
        np.random.seed(2021)
        paddle.seed(2021)

        multiprocessing.set_start_method('spawn')
        self.task_queue = multiprocessing.Queue()
        self.result_queue = multiprocessing.Queue()
        # TODO(@gexiao): Currently only supports gpu env
        expected_device_id = (
            int(os.getenv("FLAGS_selected_gpus"))
            if core.is_compiled_with_cuda()
            else 0
        )
        self.fusion_worker = FusionWorker(
            0, expected_device_id, self.task_queue, self.result_queue
        )
        self.fusion_worker.start()
        self.fusion_buffer_version = 0

        self.strategy = fleet.DistributedStrategy()
        self.strategy.hybrid_configs = {
            "sharding_degree": 2,
            "dp_degree": 1,
            "mp_degree": 1,
            "pp_degree": 1,
        }
        self.strategy.hybrid_configs[
            "sharding_configs"
        ].split_param = g_shard_split_param

        fleet.init(is_collective=True, strategy=self.strategy)
        self.data = [
            np.random.randint(
                0,
                vocab_size,
                (
                    batch_size,
                    seq_length,
                ),
            )
            for _ in range(STEPS)
        ]

        atexit.register(self.shutdown)

    def train_batch(self, batch, model, optimizer):
        output = model(batch)
        loss = output.mean()
        loss.backward()  # do backward
        optimizer.step()  # update parameters
        optimizer.clear_grad()
        return loss

    def build_optimizer(self, model, strategy=None, Optimizer="adam"):
        clip = paddle.nn.ClipGradByGlobalNorm(0.5)
        if Optimizer == "adam":
            optimizer = paddle.optimizer.AdamW(
                parameters=model.parameters(),
                learning_rate=0.001,
                weight_decay=0.00001,
                grad_clip=clip,
            )
        else:
            optimizer = paddle.optimizer.Momentum(
                learning_rate=0.001,
                parameters=model.parameters(),
                grad_clip=clip,
            )
        return optimizer

    def build_model_optimizer(self, Optimizer="adam", amp_level=None):
        hcg = fleet.get_hybrid_communicate_group()
        word_size = hcg.get_model_parallel_world_size()
        sharding_id = hcg.get_sharding_parallel_rank()
        dp_id = hcg.get_data_parallel_rank()
        rank_id = dist.get_rank()

        np_fc1 = np.random.random_sample((hidden_size, inner_size))
        np_fc2 = np.random.random_sample((inner_size, hidden_size))

        model_a = SimpleDPNet(
            vocab_size, hidden_size, inner_size, output_size, np_fc1, np_fc2
        )
        optimizer_a = self.build_optimizer(
            model_a,
            strategy=self.strategy,
            Optimizer=Optimizer,
        )

        model_b = SimpleDPNet(
            vocab_size, hidden_size, inner_size, output_size, np_fc1, np_fc2
        )
        optimizer_b = self.build_optimizer(
            model_b,
            strategy=self.strategy,
            Optimizer=Optimizer,
        )

        if amp_level is not None and amp_level == "O2":
            model_a, optimizer_a = paddle.amp.decorate(
                models=model_a,
                optimizers=optimizer_a,
                level=amp_level,
                dtype="float16",
            )
            model_b, optimizer_b = paddle.amp.decorate(
                models=model_b,
                optimizers=optimizer_b,
                level=amp_level,
                dtype="float16",
            )
            model_a = MixPrecisionLayer(model_a)
            optimizer_a = MixPrecisionOptimizer(optimizer_a)
            model_b = MixPrecisionLayer(model_b)
            optimizer_b = MixPrecisionOptimizer(optimizer_b)

        model_a = fleet.distributed_model(model_a)
        optimizer_a = fleet.distributed_optimizer(optimizer_a)

        strategy = copy.deepcopy(fleet.fleet._user_defined_strategy)
        strategy.hybrid_configs[
            "sharding_configs"
        ].enable_fuse_optimizer_states = True
        model_b = fleet.distributed_model(model_b)
        optimizer_b = fleet.distributed_optimizer(optimizer_b, strategy)

        return model_a, optimizer_a, model_b, optimizer_b

    def sharding_model(self, Optimizer, amp_level=None):
        model_a, optimizer_a, model_b, optimizer_b = self.build_model_optimizer(
            Optimizer=Optimizer, amp_level=amp_level
        )
        shard_opt_cls = (
            DygraphShardingOptimizerV2
            if g_shard_split_param
            else DygraphShardingOptimizer
        )
        self.assertTrue(isinstance(optimizer_a._inner_opt, shard_opt_cls))

        for idx in range(STEPS):
            if paddle.distributed.get_rank() == 0:
                batch_sharding = paddle.to_tensor(self.data[idx][:2])
            else:
                batch_sharding = paddle.to_tensor(self.data[idx][2:])

            loss_a = self.train_batch(batch_sharding, model_a, optimizer_a)
            loss_b = self.train_batch(batch_sharding, model_b, optimizer_b)

            for j in range(len(model_a.parameters())):
                np.testing.assert_equal(
                    model_a.parameters()[j].numpy(),
                    model_b.parameters()[j].numpy(),
                )
            if self.fusion_buffer_version != optimizer_b.fused_buffer_version:
                # merged params not supported yet
                meta_infos = (
                    optimizer_b.fused_states_accumulators_meta,
                    optimizer_b.fused_states_master_weights_meta,
                    None,
                    optimizer_b.fused_states_buffer_ipc_meta,
                )
                # step1: update meta infos
                task = (DO_FUSE_OPTIMIZER, meta_infos)
                self.task_queue.put(task)
                print(
                    f"[TX] queue.put obj type={type(task)} start_method=?",
                    flush=True,
                )
                self.fusion_buffer_version = optimizer_b.fused_buffer_version
            # step2: sync params
            self.task_queue.put((DO_SYNC_PARAM, None))
            # step3: get state dict
            self.task_queue.put((DO_RETURN_DICT, None))
            _, state_dict_b = self.result_queue.get()
            state_dict_a = optimizer_a.state_dict()

            master_weights_a = state_dict_a["master_weights"]
            master_weights_b = state_dict_b["master_weights"]
            for k, v in master_weights_b.items():
                np.testing.assert_equal(
                    v.detach().cpu().numpy(),
                    master_weights_b[k].detach().cpu().numpy(),
                )
            for k, v in state_dict_b.items():
                if k == "master_weights":
                    continue
                np.testing.assert_equal(
                    v.detach().cpu().numpy(),
                    state_dict_b[k].detach().cpu().numpy(),
                )

    def test_sharding_adam_enable_fuse_optimizer_states(self):
        if core.is_compiled_with_cuda():
            self.sharding_model(
                Optimizer="adam",
                amp_level="O2",
            )

    def shutdown(self):
        self.task_queue.put(None)
        self.fusion_worker.join()


if __name__ == "__main__":
    unittest.main()
