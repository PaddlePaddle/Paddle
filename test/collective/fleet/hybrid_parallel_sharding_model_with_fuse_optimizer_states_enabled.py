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
import unittest
from multiprocessing.reduction import DupFd

import numpy as np

print(
    "[BOOT] mp start_method=",
    multiprocessing.get_start_method(allow_none=True),
    flush=True,
)
# === BEGIN: Paddle CUDA-IPC trace hooks (paste this at the very top of your script) ===
import hashlib
import sys
import time
from functools import wraps


def _now():
    return time.strftime("%H:%M:%S")


def _rank():
    return (
        os.environ.get("PADDLE_TRAINER_ID")
        or os.environ.get("PADDLE_RANK")
        or os.environ.get("RANK")
        or "?"
    )


def _short(b):
    try:
        return hashlib.sha1(b).hexdigest()[:8]
    except Exception:
        return "NA"


print(
    f"[{_now()}][IPC-HOOK] installing hooks... pid={os.getpid()} rank={_rank()}",
    flush=True,
)

# 1) Hook reductions: _reduce_lodtensor (TX) / _rebuild_cuda_tensor (RX)
try:
    from paddle.incubate.multiprocessing import reductions as red

    _orig_reduce_lt = red._reduce_lodtensor
    _orig_rebuild_cuda = red._rebuild_cuda_tensor

    def _spy_reduce_lodtensor(lodtensor):
        place = lodtensor._place()
        print(
            f"[{_now()}][IPC-TX] rank={_rank()} pid={os.getpid()} "
            f"_reduce_lodtensor place={place}",
            flush=True,
        )
        # traceback.print_stack(limit=12, file=sys.stdout)
        rv = _orig_reduce_lt(lodtensor)
        # rv = (rebuild_fn, (cls, handle, offset, size, dtype, dims, lod, dev))
        try:
            rebuild_fn, meta = rv
            if rebuild_fn is _orig_rebuild_cuda and len(meta) >= 3:
                handle_bytes = meta[1]
                off = meta[2]
                size = meta[3]
                print(
                    f"[{_now()}][IPC-TX] rank={_rank()} pid={os.getpid()} "
                    f"cuda_handle_sha1={_short(handle_bytes)} off={off} size={size}B",
                    flush=True,
                )
        except Exception as _e:
            print(
                f"[{_now()}][IPC-TX] decode reduce meta failed: {_e}",
                file=sys.stderr,
                flush=True,
            )
        return rv

    def _spy_rebuild_cuda_tensor(
        cls, handle, offset_bytes, size, type_idx, dims, lod, device_idx
    ):
        print(
            f"[{_now()}][IPC-RX] rank={_rank()} pid={os.getpid()} "
            f"_rebuild_cuda_tensor dev={device_idx} handle_sha1={_short(handle)} off={offset_bytes} size={size}B",
            flush=True,
        )
        # # traceback.print_stack(limit=12, file=sys.stdout)
        return _orig_rebuild_cuda(
            cls, handle, offset_bytes, size, type_idx, dims, lod, device_idx
        )

    red._reduce_lodtensor = _spy_reduce_lodtensor
    red._rebuild_cuda_tensor = _spy_rebuild_cuda_tensor
    print(f"[{_now()}][IPC-HOOK] reductions hooks installed", flush=True)
except Exception as e:
    print(
        f"[{_now()}][IPC-HOOK] reductions hook failed: {e}",
        file=sys.stderr,
        flush=True,
    )

# 2) Hook pybind: DenseTensor._share_cuda / _new_shared_cuda
try:
    import paddle
    from paddle.base import core

    _orig_share_cuda = core.DenseTensor._share_cuda
    _orig_new_shared_cuda = core.DenseTensor._new_shared_cuda
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
            fd_safely = DupFd(fd)  # ★★ 核心：包装 fd ★★
            meta = (fd_safely, off, size, dtype, dims, lod, dev)
            print(
                f"[{_now()}][IPC-TX] rank={_rank()} pid={os.getpid()} "
                f"_share_vmm dev={dev} fd(type)={type(fd_safely).__name__} off={off} size={size}B",
                flush=True,
            )
            print(
                f"[TX] before put: type(fd)={type(fd).__name__}  off={off}  size={size}  dev={dev}",
                flush=True,
            )
            # 可选：打印调用栈
            # traceback.print_stack(limit=10, file=sys.stdout)
        except Exception as e:
            print(
                f"[{_now()}][IPC-TX] _share_vmm meta decode failed: {e}",
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
                f"[{_now()}][IPC-RX] rank={_rank()} pid={os.getpid()} "
                f"_new_shared_vmm dev={dev} fd(int)={fd} off={off} size={size}B",
                flush=True,
            )
            # 可选：打印调用栈
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
                f"[{_now()}][IPC-RX] _new_shared_vmm precheck failed: {e}",
                file=sys.stderr,
                flush=True,
            )
        return _orig_new_shared_vmm(meta)

    # ★★ 正确绑定（你之前错绑到了 _share_cuda/_new_shared_cuda）★★
    core.DenseTensor._share_vmm = _spy_share_vmm
    core.DenseTensor._new_shared_vmm = _spy_new_shared_vmm

    print(
        f"[{_now()}][IPC-HOOK] VMM hooks installed (pid={os.getpid()} rank={_rank()})",
        flush=True,
    )

    @wraps(_orig_share_cuda)
    def _spy_share_cuda(self):
        meta = _orig_share_cuda(self)
        try:
            handle, off, size, dtype, dims, lod, dev = meta
            print(
                f"[{_now()}][IPC-TX] rank={_rank()} pid={os.getpid()} "
                f"_share_cuda dev={dev} handle_sha1={_short(handle)} off={off} size={size}B",
                flush=True,
            )
            # traceback.print_stack(limit=12, file=sys.stdout)
        except Exception as _e:
            print(
                f"[{_now()}][IPC-TX] _share_cuda meta decode failed: {_e}",
                file=sys.stderr,
                flush=True,
            )
        return meta

    @wraps(_orig_new_shared_cuda)
    def _spy_new_shared_cuda(meta):
        try:
            handle, off, size, dtype, dims, lod, dev = meta
            print(
                f"[{_now()}][IPC-RX] rank={_rank()} pid={os.getpid()} "
                f"_new_shared_cuda dev={dev} handle_sha1={_short(handle)} off={off} size={size}B",
                flush=True,
            )
            # traceback.print_stack(limit=12, file=sys.stdout)
        except Exception as _e:
            print(
                f"[{_now()}][IPC-RX] _new_shared_cuda meta decode failed: {_e}",
                file=sys.stderr,
                flush=True,
            )
        return _orig_new_shared_cuda(meta)

    core.DenseTensor._share_cuda = _spy_share_cuda
    core.DenseTensor._new_shared_cuda = _spy_new_shared_cuda
    print(
        f"[{_now()}][IPC-HOOK] core DenseTensor CUDA-IPC hooks installed",
        flush=True,
    )
except Exception as e:
    print(
        f"[{_now()}][IPC-HOOK] core hook failed: {e}",
        file=sys.stderr,
        flush=True,
    )

# 3) Hook multiprocessing.Queue put/get (看到谁在跨进程传对象)
try:
    import multiprocessing.queues as mpq

    _orig_put = mpq.Queue.put
    _orig_get = mpq.Queue.get

    @wraps(_orig_put)
    def _spy_put(self, obj, *args, **kwargs):
        print(
            f"[{_now()}][MP-PUT] rank={_rank()} pid={os.getpid()} type={type(obj).__name__}",
            flush=True,
        )
        # traceback.print_stack(limit=12, file=sys.stdout)
        return _orig_put(self, obj, *args, **kwargs)

    @wraps(_orig_get)
    def _spy_get(self, *args, **kwargs):
        obj = _orig_get(self, *args, **kwargs)
        print(
            f"[{_now()}][MP-GET] rank={_rank()} pid={os.getpid()} type={type(obj).__name__}",
            flush=True,
        )
        # traceback.print_stack(limit=12, file=sys.stdout)
        return obj

    mpq.Queue.put = _spy_put
    mpq.Queue.get = _spy_get
    print(f"[{_now()}][IPC-HOOK] mp.Queue hooks installed", flush=True)
except Exception as e:
    print(
        f"[{_now()}][IPC-HOOK] mp.Queue hook failed: {e}",
        file=sys.stderr,
        flush=True,
    )

# 4) （可选）Hook 分布式广播（对象/张量），若你的路径使用它们可以看到栈
try:
    import paddle.distributed as dist

    if hasattr(dist, "broadcast"):
        _orig_b = dist.broadcast

        @wraps(_orig_b)
        def _spy_b(*args, **kwargs):
            tensor = args[0] if args else kwargs.get("tensor", None)
            src = kwargs.get("src", args[1] if len(args) > 1 else None)
            group = kwargs.get("group", None)
            shape = (
                tuple(getattr(tensor, "shape", []))
                if tensor is not None
                else None
            )
            dtype = (
                getattr(tensor, "dtype", None) if tensor is not None else None
            )
            place = (
                getattr(
                    getattr(tensor, "place", None),
                    "_typename",
                    str(getattr(tensor, "place", None)),
                )
                if tensor is not None
                else None
            )
            print(
                f"[{_now()}][DIST] broadcast src={src} group={group} shape={shape} dtype={dtype} place={place}",
                flush=True,
            )
            # traceback.print_stack(limit=12, file=sys.stdout)
            return _orig_b(*args, **kwargs)

        dist.broadcast = _spy_b

    if hasattr(dist, "broadcast_object_list"):
        _orig_bol = dist.broadcast_object_list

        @wraps(_orig_bol)
        def _spy_bol(*args, **kwargs):
            object_list = args[0] if args else kwargs.get("object_list", None)
            src = kwargs.get("src", args[1] if len(args) > 1 else None)
            group = kwargs.get("group", None)

            def _slen(o):
                try:
                    return len(o)
                except Exception:
                    return None

            types = [type(o).__name__ for o in (object_list or [])]
            sizes = [_slen(o) for o in (object_list or [])]
            print(
                f"[{_now()}][DIST] broadcast_object_list src={src} group={group} "
                f"types={types} sizes={sizes}",
                flush=True,
            )
            # traceback.print_stack(limit=12, file=sys.stdout)
            return _orig_bol(*args, **kwargs)

        dist.broadcast_object_list = _spy_bol
    print(f"[{_now()}][IPC-HOOK] dist broadcast hooks installed", flush=True)
except Exception as e:
    print(
        f"[{_now()}][IPC-HOOK] dist hook failed: {e}",
        file=sys.stderr,
        flush=True,
    )

print(
    f"[{_now()}][IPC-HOOK] all hooks installed (pid={os.getpid()} rank={_rank()})",
    flush=True,
)
# === END: Paddle CUDA-IPC trace hooks ===


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

"""
if hasattr(dist, "broadcast"):
    _orig_b = dist.broadcast

    @wraps(_orig_b)
    def _spy_b(*args, **kwargs):
        # 尽力拿到几个关键字段做日志
        tensor = args[0] if args else kwargs.get("tensor", None)
        src    = kwargs.get("src", args[1] if len(args) > 1 else None)
        group  = kwargs.get("group", None)
        try:
            shape = tuple(getattr(tensor, "shape", []))
            dtype = getattr(tensor, "dtype", None)
            place = getattr(getattr(tensor, "place", None), "_typename", str(getattr(tensor, "place", None)))
        except Exception:
            shape = dtype = place = None

        print(f"[spy] dist.broadcast src={src} group={group} shape={shape} dtype={dtype} place={place}")
        # traceback.print_stack(limit=5, file=sys.stdout)
        return _orig_b(*args, **kwargs)

    dist.broadcast = _spy_b

if hasattr(dist, "broadcast_object_list"):
    _orig_bol = dist.broadcast_object_list

    @wraps(_orig_bol)
    def _spy_bol(*args, **kwargs):
        # object_list 可能是第一个位置参数或关键字
        if args:
            object_list = args[0]
        else:
            object_list = kwargs.get("object_list", None)
        src   = kwargs.get("src", args[1] if len(args) > 1 else None)
        group = kwargs.get("group", None)

        # 打印每个对象的类型和可用长度
        def _safe_len(o):
            try:
                return len(o)
            except Exception:
                return None

        types = [type(o).__name__ for o in (object_list or [])]
        sizes = [_safe_len(o) for o in (object_list or [])]
        print(f"[spy] broadcast_object_list src={src} group={group} types={types} sizes={sizes}")
        # traceback.print_stack(limit=5, file=sys.stdout)
        return _orig_bol(*args, **kwargs)

    dist.broadcast_object_list = _spy_bol
"""


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
