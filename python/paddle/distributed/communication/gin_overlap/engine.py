# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

"""gin_overlap 的通信原语：NCCL GIN 单边通信代替 sharding stage1 梯度的 reduce_scatter。

  专家桶（moe_sharding 组 expert_group_size 个 rank）：反向里桶满 → 在 DeepEP 不跑的窗口里 GIN put 给其余 peer（组内序号）
  dense 桶（sharding 组 = num_nodes 节点 × ranks_per_node 卡，要求组内 rank 升序、每节点成员数一致）：
      机内 CE 推列（窗口 A/B） → 机内求和（pre 窗口） → 同卡号机间 put（窗口 A/B） → 本卡分片最终求和（pre 窗口）

两个通信组各一套资源：comm（都从 Paddle 已有的 ProcessGroup 取，
paddle/fluid/distributed/collective/gin_overlap/paddle_comm.cc）、devcomm、send / recv / signal
显存 window 和一块 host window；组内寻址一律用组内序号。
接收池（dense 组的 recv window 分机内 / 机间两段、专家组的 recv window 一段，每段是一个环，所以多个桶能同时在途）、
远端条件由 CPU 直接读 host window 判断后再交给 GPU（判断在 C++ 侧的放行层
paddle/fluid/distributed/collective/gin_overlap/kernels/release.cu）。
各桶 grad_storage 换成本组 send window 的切片（零打包）：切片由 C++ 侧 paddle::from_blob 包成张量
（lib.send_window_tensor，见 gin_overlap/paddle_tensor.cc）。

这个文件只有「布局 + 怎么交给 C++ 侧」这一半：窗口 / 接收池 / host window 的布局、两个通信组的建立和
校验、交给放行层的每桶参数表、2D 桶 gather 的表，以及初始化日志。**什么时候**推进哪个阶段（哪个窗口试哪些
阶段、尾部交接、桶队列）不在这里 —— 那是训练框架的事，在 PaddleFleet 的
paddlefleet/distributed/gin_overlap/{scheduler,backend}.py；和 Paddle 优化器对象打交道的那部分
（桶的 grad_storage 别名、梯度系数怎么乘）也在那边（它的 engine.py，继承这里的 GinOverlapPrimitive）。
"""

import ctypes
import os
import time
from collections import Counter

import paddle
import paddle.distributed as dist

# 这一层的 C++ 实现（paddle.base.core.gin_overlap，源码在
# paddle/fluid/distributed/collective/gin_overlap/）只在满足条件的构建里才编进去：
# WITH_GPU + WITH_NCCL + NCCL >= 2.30.7（要它的设备 LSA/GIN API）+ CUDA arch 含 9.0 / 10.0，
# 见 paddle/fluid/distributed/collective/CMakeLists.txt 的门控。不满足时
# paddle.base.core 上就没有 gin_overlap 这个属性（而不是一个半截模块），报错提示里用这一行。
_BUILD_HINT = (
    "这个 paddlepaddle 没有编 gin_overlap：要 WITH_GPU + WITH_NCCL、NCCL >= 2.30.7 "
    "（带 nccl_device.h）、CUDA arch 含 9.0 或 10.0"
)

# 固定参数（都影响不到数值）
# put kernel 的 block 数（只在这里定义一份，release_setup 交给放行层）
PUT_BLOCKS = 64
GIN_CONTEXTS = 8  # GIN context 数 hint
TIMEOUT_S = 300.0  # 等对端超时就报错，不挂住作业
NUM_WINDOW_MARKS = 1024  # 窗口 mark 池大小（每步约 130 个）
# 尾部 mark 环（lib.tail_mark，和窗口 mark 池分开）：每步最多 = 1（交接）+ 尾部之后 ready 的桶数（≤ 桶数）
#   + 尾部里又过的 combine 数，环大小 = 桶数 + 这个余量；一步内用完就报错，不绕回 —— 绕回会覆盖掉尾部线程正在等的 event
TAIL_EXTRA_MARKS = 1024
LOG_EVERY = 100  # 每多少步打一行累计流量 / 调度汇总（只是日志，不影响数值）
# 通信组编号（和 C++ 侧 gin_overlap/gin_engine.h 的 GROUP_DENSE / GROUP_MOE 一致，device_engine() 里核对）：
#   dense 桶走 sharding 组、专家桶走 moe_sharding 组，各自一套 comm / devcomm / send / recv / signal / host window
GROUP_DENSE, GROUP_MOE = 0, 1
GROUP_NAME = {GROUP_DENSE: "sharding", GROUP_MOE: "moe_sharding"}

_DEVICE_ENGINE = None


def device_engine():
    """设备侧那个引擎对象：paddle.base.core.gin_overlap.GradCommEngine，进程内只有一个
    （所有 window / 侧流 / event / 放行层和它的尾部线程都挂在它上面；Python 侧只管布局和编排）。
    第一次用到才取、才建对象：这个子模块只在满足条件的构建里才有（见 _BUILD_HINT），拿不到时要给
    说明，而不是在 import 这个文件时就炸；顺带核对通信组编号和 NCCL 版本（编译期 vs 运行时）。"""
    global _DEVICE_ENGINE
    if _DEVICE_ENGINE is None:
        from paddle.base import core

        lib = getattr(core, "gin_overlap", None)
        if lib is None:
            raise RuntimeError(
                f"没有 paddle.base.core.gin_overlap：{_BUILD_HINT}"
            )
        if (lib.GROUP_DENSE, lib.GROUP_MOE) != (GROUP_DENSE, GROUP_MOE):
            raise RuntimeError(
                f"gin_overlap: C++ 侧的通信组编号 ({lib.GROUP_DENSE}, {lib.GROUP_MOE}) "
                f"和 Python 侧 ({GROUP_DENSE}, {GROUP_MOE}) 不一致（改一边忘另一边）"
            )
        eng = lib.GradCommEngine()
        rt, bt = eng.nccl_runtime_version(), eng.nccl_build_version()
        if rt != bt:
            raise RuntimeError(
                f"gin_overlap: C++ 侧是按 NCCL {bt} 编的，但进程里加载的是 NCCL {rt} —— "
                "设备 API 的结构布局是编进 cubin 的，两者必须是同一个 libnccl"
            )
        _DEVICE_ENGINE = eng
    return _DEVICE_ENGINE


def check_cuda(rc, what):
    if rc != 0:
        raise RuntimeError(
            f"gin_overlap {what} 失败: {device_engine().error_string(rc)}"
        )


def log(msg):
    if dist.get_rank() == 0:
        print(f"[gin_overlap] {msg}", flush=True)


def _align(n, a=256):
    return (n + a - 1) // a * a


def _gib(x):
    return format(x / 2**30, ".3f")


def local_size():
    """每台机器的卡数：PADDLE_LOCAL_SIZE，取不到退回可见设备数（做法同
    python/paddle/distributed/fleet/meta_optimizers/muon_sharding_optimizer.py:97-118
    的 get_gpus_per_node）。"""
    v = os.getenv("PADDLE_LOCAL_SIZE", "").strip()
    if v:
        try:
            n = int(v)
        except ValueError:
            n = 0
        if n > 0:
            return n
    return max(1, paddle.device.cuda.device_count())


class GinOverlapPrimitive:
    def __init__(
        self,
        world,
        moe_group,
        moe_numels,
        dense_numels,
        release_order,
        dense_recv_pool_gb,
        moe_recv_pool_gb,
    ):
        """集合操作：world（dense 的 sharding 组）内所有 rank 同时调用；有专家桶时专家组（moe_sharding）内所有 rank
        也同时调用。release_order：(dense 桶固定放行顺序, 专家桶固定放行顺序)，调用方给（PaddleFleet 的
        backend._release_order）。dense_recv_pool_gb / moe_recv_pool_gb：两个接收池的大小（GiB），
        调用方的配置里来（PaddleFleet 的 gin_overlap/config.py）。
        dense 桶走 sharding 组、专家桶走 moe_sharding 组：两个 comm、两个 devcomm，每组自己的 send / recv / signal /
        host window（recv window 的分段见 _layout_recv_pool，host window 里几块计数区的位置见 _layout_host_windows）；
        组内寻址一律用组内序号。"""
        device_engine()  # 第一次用到 C++ 侧：这里就取 + 核对通信组编号 / NCCL 版本，别等到后面
        self.dense_recv_pool_gb = dense_recv_pool_gb
        self.moe_recv_pool_gb = moe_recv_pool_gb
        self.world = world
        self.moe_comm_group = moe_group
        self.group_size, self.group_rank = world.nranks, world.rank
        t_init = [time.perf_counter()]

        self.moe_bucket_numels = list(moe_numels)
        self.expert_group_size = self.expert_group_rank = 0
        if self.moe_bucket_numels:
            assert moe_group is not None, "有专家桶但没有专家通信组"
            self.expert_group_size, self.expert_group_rank = (
                moe_group.nranks,
                moe_group.rank,
            )
            for n in self.moe_bucket_numels:
                assert n % self.expert_group_size == 0, (
                    f"专家桶大小 {n} 不能被组大小 {self.expert_group_size} 整除"
                )
        self.dense_bucket_numels = list(dense_numels)
        for n in self.dense_bucket_numels:
            assert n % self.group_size == 0, (
                f"dense 桶大小 {n} 不能被组大小 {self.group_size} 整除"
            )

        # 拓扑要求：两个组各自要求「rank 列表升序 + 每节点成员数一致」⇒ 组内序号按机器分块，
        #   rank_in_node = 组内序号 % ranks_per_node、node_index = 组内序号 // ranks_per_node。
        #   各自 devcomm 的 lsaRank / lsaSize 在下面和这里推算的值核对，不一致报错（不静默算错）。
        self.ranks_per_node, self._local_size = self._topology(
            world, "sharding"
        )
        self.num_nodes = self.group_size // self.ranks_per_node
        self.rank_in_node = self.group_rank % self.ranks_per_node
        self.node_index = self.group_rank // self.ranks_per_node
        self.moe_ranks_per_node, self.moe_num_nodes = 1, 1
        self.moe_rank_in_node = self.moe_node_index = 0
        if self.moe_bucket_numels:
            self.moe_ranks_per_node, _ = self._topology(
                moe_group, "moe_sharding"
            )
            self.moe_num_nodes = (
                self.expert_group_size // self.moe_ranks_per_node
            )
            self.moe_rank_in_node = (
                self.expert_group_rank % self.moe_ranks_per_node
            )
            self.moe_node_index = (
                self.expert_group_rank // self.moe_ranks_per_node
            )

        # 每组自己的 send / recv / signal window，组内桶之间用字节偏移（两组各自从 0 排）
        self.moe_send_offsets, moe_send_bytes = self._send_layout(
            self.moe_bucket_numels
        )
        self.dense_send_offsets, dense_send_bytes = self._send_layout(
            self.dense_bucket_numels
        )
        num_moe, num_dense = (
            len(self.moe_bucket_numels),
            len(self.dense_bucket_numels),
        )
        # 每桶两格：[0] 机间 / 专家 GIN signal，[1] 不再使用
        self.moe_signal_offsets = [16 * i for i in range(num_moe)]
        self.dense_signal_offsets = [16 * i for i in range(num_dense)]
        recv_bytes = self._layout_recv_pool(
            release_order, moe_group
        )  # (dense recv 字节, 专家 recv 字节)

        dev = paddle.distributed.ParallelEnv().device_id
        self._set_comms(dev, world, moe_group)
        t_init.append(time.perf_counter())
        signal_bytes = (
            self._layout_host_windows()
        )  # 两组的 signal / host window 布局（返回各组 signal window 字节）
        p = self.recv_pool
        # 各 rank 按自己的大小注册（NCCL 允许 window 大小不同，只要求对端访问不越界）
        self._win = {}
        self._hw = {}
        self._setup_group(
            GROUP_DENSE, "sharding", dense_send_bytes, recv_bytes[0], signal_bytes[0], p["host_bytes"][0],
            self.group_size, self.group_rank, self.ranks_per_node, self.rank_in_node, self.num_nodes, self.node_index,
            world,
        )  # fmt: skip
        if num_moe:
            self._setup_group(
                GROUP_MOE, "moe_sharding", moe_send_bytes, recv_bytes[1], signal_bytes[1], p["host_bytes"][1],
                self.expert_group_size, self.expert_group_rank, self.moe_ranks_per_node, self.moe_rank_in_node,
                self.moe_num_nodes, self.moe_node_index, moe_group,
            )  # fmt: skip
        t_init.append(time.perf_counter())
        for n in self.dense_bucket_numels:
            assert (n // self.group_size) % 4 == 0, (
                f"dense 桶每块 numel 要是 4 的倍数：{n // self.group_size}"
            )
        self._configure(num_dense, num_moe)
        t_init.append(time.perf_counter())
        self.moe_offsets = list(
            zip(self.moe_send_offsets, self.moe_signal_offsets)
        )
        self.dense_offsets = list(
            zip(self.dense_send_offsets, self.dense_signal_offsets)
        )
        self._init_times = t_init

    @staticmethod
    def _send_layout(numels):
        """本组 send window：每个桶一段，256B 对齐。返回 ({桶号: 字节偏移} 的列表, 总字节)。"""
        offsets, total = [], 0
        for b in [n * 4 for n in numels]:
            offsets.append(total)
            total += _align(b)
        return offsets, total

    def _set_comms(self, dev, world, moe_group):
        """复用 Paddle 已经建好的 ncclComm（不自己 ncclCommInitRank）：把两个组的 ring id 交给 C++ 侧，它照 DeepEP 的做法
        从 paddle::distributed::ProcessGroupMapFromGid 取 ProcessGroup → ProcessGroupNCCL::GetOrCreateCommContext
        → NCCLCommContext::GetNcclComm（paddle/fluid/distributed/collective/gin_overlap/paddle_comm.cc）。

        ring id 用 Paddle 通信组对象的 `.id`（同目录 `../group.py:78` 的 `Group.id` 属性，
        就是 C++ 侧注册进 ProcessGroupMapFromGid 的 gid；DeepEP 也是把 `group.id` 传给 C++，见
        `../deep_ep/buffer.py:109`）。C++ 侧会把回读的组大小 / 组内序号和这里传的核对，填错 ring id 会报错。
        拿不到（符号缺失 / 不是 ProcessGroupNCCL / Paddle 太旧）就报错，**不**回退到自己建 comm。

        时机：`GetOrCreateCommContext` 在这个组还没建 NCCL comm 时会现建，我们在建引擎（构建 sharding 优化器）时调它，
        可能比 Paddle 自己首次用这个组更早。"""
        rc, err = device_engine().set_comms(
            int(world.id),
            int(moe_group.id)
            if (moe_group is not None and self.moe_bucket_numels)
            else -1,
            dev,
            self.group_size,
            self.group_rank,
            self.expert_group_size,
            self.expert_group_rank,
        )
        if rc:
            raise RuntimeError(
                f"gin_overlap: 取 Paddle 已有的 ncclComm 失败：{err or device_engine().error_string(rc)}"
            )

    def _setup_group(self, gid, what, send_bytes, recv_bytes, signal_bytes, host_bytes, size, rank, rpn, ml, nn, mn,
                     group):  # fmt: skip
        """一个组的 window：alloc_windows → host window（集合注册，必须在 create_dev_comm 之前）→ create_dev_comm
        → 校验 devcomm 的 lsaRank / lsaSize（组内一起判）→ 把组内机器分块交给 C++。"""
        lib = device_engine()
        self._win[gid] = [
            max(send_bytes, 4096),
            max(recv_bytes, 4096),
            _align(signal_bytes, 4096),
        ]
        check_cuda(
            lib.alloc_windows(gid, *self._win[gid]),
            f"ncclMemAlloc/window({what})",
        )
        rc, ptr, nbytes = lib.alloc_host_counters(gid, host_bytes)
        check_cuda(rc, f"alloc_host_counters({what})")
        if nbytes < host_bytes:  # C++ 侧按粒度向上取整，正常不会触发
            raise RuntimeError(
                f"gin_overlap: {what} 组的 host window {nbytes} B 小于需要的 {host_bytes} B"
            )
        self.recv_pool["host_window_bytes"][gid] = nbytes
        # host window 的 u64 视图（ctypes 数组，每次下标访问都是一次 8 字节对齐的内存读，不缓存）。只读：
        #   CPU 不能写正在被网卡 atomic 的计数
        self._hw[gid] = (ctypes.c_ulonglong * (nbytes // 8)).from_address(ptr)
        rc, nctx, lsa_rank, lsa_size = lib.create_dev_comm(gid, GIN_CONTEXTS)
        check_cuda(rc, f"ncclDevCommCreate({what})")
        self.recv_pool["nctx"][gid] = nctx
        self._check_devcomm(
            what, group, size, rank, rpn, ml, nn, mn, nctx, lsa_rank, lsa_size
        )
        check_cuda(
            lib.set_group_topology(gid, nn, mn), f"set_group_topology({what})"
        )

    # ------------------------------------------------------------------ 拓扑
    def _topology(self, group, what):
        """返回 (组内每节点成员数 ranks_per_node, 每机卡数 local_size)。要求组内 rank 升序、每个节点的成员数一致
        （tp / pp > 1 时组是全局 rank 的子集，成员数 < 每机卡数，所以不要求两者相等）。不满足直接报错。
        dense（sharding）组和专家（moe_sharding）组各判一次。"""
        ranks = list(group.ranks)
        size = group.nranks
        ls = local_size()
        if ranks != sorted(ranks):
            raise RuntimeError(
                f"gin_overlap: {what} 组的 rank 列表不是升序（{ranks[:16]}…），算不出机器分块"
            )
        per_node = Counter(r // ls for r in ranks)
        sizes = set(per_node.values())
        if len(sizes) != 1:
            raise RuntimeError(
                f"gin_overlap: {what} 组在各节点上的成员数不一致（每机卡数 PADDLE_LOCAL_SIZE={ls}，"
                f"各节点成员数 {dict(per_node)}），组内的取模取余不成立"
            )
        rpn = sizes.pop()
        if rpn < 1 or size % rpn:
            raise RuntimeError(
                f"gin_overlap: {what} 组大小 {size} 不是每节点成员数 {rpn} 的整数倍"
            )
        return rpn, ls

    def _check_devcomm(
        self, what, group, size, rank, rpn, ml, nn, mn, nctx, lsa_rank, lsa_size
    ):
        """devcomm 的 lsaRank / lsaSize 要和 _topology 推算的一致（组内一起判，组里任一 rank 不满足都报错）。"""
        ok = int(
            nctx >= 1
            and lsa_size == rpn
            and lsa_rank == ml
            and rpn <= 16
            and nn * rpn == size
        )
        t = paddle.to_tensor([ok, -rpn, rpn], dtype="int32")
        dist.all_reduce(t, op=dist.ReduceOp.MIN, group=group)
        if int(t[0]) != 1 or int(-t[1]) != int(t[2]):
            raise RuntimeError(
                f"gin_overlap: {what} 组的 GIN devcomm 不可用或组内机器分块和 devcomm 的 LSA team 不一致"
                f"（ctx={nctx} lsaRank={lsa_rank} lsaSize={lsa_size}，按 rank 列表推算 rank_in_node={ml} "
                f"ranks_per_node={rpn} node_index={mn} num_nodes={nn}，每机卡数 PADDLE_LOCAL_SIZE={self._local_size}，"
                f"组内序号 {rank}/{size}）"
            )

    # ------------------------------------------------------------------ 接收池（环形映射、多桶同时在途）
    def _layout_recv_pool(self, order, moe_group):
        """三段布局：每段是一个环，桶按固定顺序（机内段 / 机间段按 dense 顺序，专家段按专家顺序）依次排，放不下就绕回 0
        （_recv_pool_ring）；每个桶在段里的偏移和「前任距离」d 在这里算好。各 rank 必须一致：dense 两段全 world、专家段组内 crc 校验。
        每段下限 = 本段最大桶的槽大小（256B 对齐后向上取整到 4KB），dense 两段全 world 取 MAX、专家段组内取 MAX。
        大小：dense_recv_pool_gb = 机内段 + 机间段（按两段下限的比例分，各自 4KB 向下对齐，仍须 ≥ 各自下限），
        moe_recv_pool_gb = 专家段（4KB 向下对齐）。机内段 / 机间段在 dense 组的 recv window（依次排），专家段单独在
        专家组的 recv window（从 0 开始）。返回 (dense recv window 字节数, 专家 recv window 字节数)。"""
        import statistics
        import zlib

        nd, nm = len(self.dense_bucket_numels), len(self.moe_bucket_numels)
        d_order, m_order = (
            [int(i) for i in order[0]],
            [int(i) for i in order[1]],
        )
        if sorted(d_order) != list(range(nd)) or sorted(m_order) != list(
            range(nm)
        ):
            raise RuntimeError(
                f"gin_overlap: 放行顺序不是桶号的排列（dense {nd} 个 / 专家 {nm} 个）"
            )
        L, Nn = self.ranks_per_node, self.num_nodes
        N = max(self.expert_group_size, 1)
        dm = [n // self.group_size for n in self.dense_bucket_numels]
        mm = [n // N for n in self.moe_bucket_numels]
        seg = [  # 每个桶在各段里的槽大小
            [
                _align((L - 1) * Nn * m * 4) if L > 1 else 0 for m in dm
            ],  # 机内段：只放另外 L-1 张卡推来的行
            [
                _align(Nn * m * 4) if Nn > 1 else 0 for m in dm
            ],  # 机间段：Nn 格，自己那格空着
            [
                _align(N * m * 4) if N > 1 else 0 for m in mm
            ],  # 专家段：N 格，自己那格空着
        ]
        big = [
            max(range(len(s)), key=lambda i: s[i]) if s else -1 for s in seg
        ]  # 本卡各段最大的桶
        lb_loc = [_align(max(s, default=0), 4096) for s in seg]
        t = paddle.to_tensor(
            lb_loc[:2], dtype="int64"
        )  # dense 两段：全 world 取 MAX
        dist.all_reduce(t, op=dist.ReduceOp.MAX, group=self.world)
        tm = paddle.to_tensor(
            lb_loc[2:], dtype="int64"
        )  # 专家段：专家组内取 MAX
        if moe_group is not None:
            dist.all_reduce(tm, op=dist.ReduceOp.MAX, group=moe_group)
        lb = [int(t[0]), int(t[1]), int(tm[0])]
        names = ("dense 机内段", "dense 机间段", "专家段")
        descs = [
            f"{names[i]}下限 {_gib(lb[i])} GiB（本卡最大桶 {'dense' if i < 2 else '专家'}#{big[i]} 槽 {_gib(lb_loc[i])} GiB）"
            for i in range(3)
        ]
        P = self._recv_pool_sizes(lb, descs)
        off_i, d_i = self._recv_pool_ring(d_order, seg[0], P[0])
        off_n, d_n = self._recv_pool_ring(d_order, seg[1], P[1])
        off_m, d_m = self._recv_pool_ring(m_order, seg[2], P[2])

        def same(obj, group):
            h = zlib.crc32(repr(obj).encode()) & 0x7FFFFFFF
            v = paddle.to_tensor([h, -h], dtype="int64")
            dist.all_reduce(v, op=dist.ReduceOp.MAX, group=group)
            return int(v[0]) == -int(v[1])

        # CE 写对端机内段、dense put 写对端机间段、专家 put 写对端专家段，偏移都取自本卡的表 ⇒ 表必须和对端一致
        ok = same(
            (
                P[0],
                P[1],
                L,
                Nn,
                d_order,
                seg[0],
                seg[1],
                off_i,
                d_i,
                off_n,
                d_n,
            ),
            self.world,
        )
        if moe_group is not None:
            ok = same((P[2], N, m_order, seg[2], off_m, d_m), moe_group) and ok
        v = paddle.to_tensor([int(ok)], dtype="int32")
        dist.all_reduce(v, op=dist.ReduceOp.MIN, group=self.world)
        if int(v[0]) != 1:
            raise RuntimeError(
                "gin_overlap: 各 rank 的接收池环形布局（段大小 / 固定顺序 / 偏移 / 前任）不一致"
                "（dense 要全 world 一致，专家要组内一致）；检查各 rank 的 dense_recv_pool_gb / moe_recv_pool_gb 和桶划分"
            )

        def fit(
            i,
        ):  # 按本段桶槽大小的中位数，这一段能同时放几个桶（只用于日志）
            s = [x for x in seg[i] if x > 0]
            if not s or not P[i]:
                return 0.0, 0
            med = statistics.median(s)
            return med, int(P[i] // med)

        self.recv_pool = {
            "total": sum(P), "lb": lb, "P": P, "desc": "；".join(descs),
            "big": [(big[i], lb_loc[i]) for i in range(3)], "fit": [fit(i) for i in range(3)],
            # 三段的起点：机内段 / 机间段在 dense 组的 recv window，专家段在专家组的 recv window（各自从 0 排）
            "base": [0, P[0], 0],
            "offsets": (off_i, off_n, off_m),  # 每段 {桶号: 段内字节偏移}（槽大小为 0 的桶不在表里）
            "distance": (d_i, d_n, d_m),  # 每段 {桶号: 前任距离 d}（1..n；n = 前任是自己上一步那份）
            "dense_order": d_order, "moe_order": m_order,  # 固定放行顺序（交给放行层，位置就是在这里面的下标）
            "buckets": None,  # bind_buckets：(专家桶, dense 桶)，最终求和时读各桶的 _gin_post_reduce_scale
            "host_bytes": [0, 0],  # _layout_host_windows 填：两组 host window 各需要多少字节
            "host_window_bytes": {},  # _setup_group 填：实际大小（按 HOST_NUMA 粒度取整）
            "nctx": {},  # _setup_group 填：两组各自的 GIN ctx 数
        }  # fmt: skip
        return P[0] + P[1], P[2]

    def _recv_pool_sizes(self, lb, descs):
        """三段的实际大小（字节，4KB 的倍数）。lb = 三段下限（4KB 的倍数），descs = 报错 / 日志用的下限说明。"""
        gd, gm = self.dense_recv_pool_gb, self.moe_recv_pool_gb
        P = list(lb)
        Sd = lb[0] + lb[1]
        if Sd > 0:
            Td = int(gd * 2**30)
            if Td < Sd:
                raise RuntimeError(
                    f"dense_recv_pool_gb={gd:g} 太小：dense 机内段 + 机间段下限之和 {_gib(Sd)} GiB（{Sd} B）；"
                    f"{descs[0]}；{descs[1]}"
                )
            P[0] = (lb[0] * Td // Sd) // 4096 * 4096  # 按两段下限的比例分
            P[1] = (lb[1] * Td // Sd) // 4096 * 4096
            for i in (0, 1):
                if (
                    P[i] < lb[i]
                ):  # lb 是 4KB 的倍数且 Td ≥ Sd 时不会发生，留着防算错
                    raise RuntimeError(
                        f"dense_recv_pool_gb={gd:g}：按比例分后{descs[i]}不满足（分到 {_gib(P[i])} GiB）"
                    )
        if lb[2] > 0:
            Tm = int(gm * 2**30) // 4096 * 4096
            if Tm < lb[2]:
                raise RuntimeError(f"moe_recv_pool_gb={gm:g} 太小：{descs[2]}")
            P[2] = Tm
        return P

    @staticmethod
    def _recv_pool_ring(order, seg, P):
        """环形映射：每个桶整块占一段，按 order 依次排，cur + size > P 就绕回 0（尾部留空）。每步从 0 开始排，所以每步布局相同。
        前任 = 新槽覆盖的区间里、上一个占用者中序号最大的（序号 = 步 × n + 位置；接收方按固定顺序腾空，序号小的先腾空）。
        维护「每段区间当前占用者」精确算重叠，模拟两步、记第 2 步的结果 d = 本桶序号 - 前任序号（1..n；n = 前任是自己上一步那份）。
        第 0 步里前任若落在「上一步」，运行时目标 ≤ 0，不等（_credit_target 取 max(0, ·)）。
        返回 ({桶: 偏移}, {桶: d})；槽大小为 0 的桶不在表里（这一段不用，credit 也不等）。"""
        n, off, cur = len(order), {}, 0
        for b in order:
            sz = seg[b]
            if sz <= 0:
                continue
            if sz > P:  # P ≥ 下限 ≥ 最大槽，正常不会发生
                raise RuntimeError(
                    f"gin_overlap: 桶 {b} 的槽 {sz} B 大于段大小 {P} B"
                )
            if cur + sz > P:
                cur = 0
            off[b] = cur
            cur += sz
        occ, d = [], {}  # occ：互不重叠的 (起, 止, 占用者序号)
        for s in (0, 1):
            for k, b in enumerate(order):
                if b not in off:
                    continue
                a, e, q = off[b], off[b] + seg[b], s * n + k
                keep, pq = [], -1
                for x, y, o in occ:
                    if y <= a or x >= e:
                        keep.append((x, y, o))
                        continue
                    pq = max(pq, o)
                    if x < a:
                        keep.append((x, a, o))
                    if y > e:
                        keep.append((e, y, o))
                keep.append((a, e, q))
                occ = sorted(keep)
                if s == 1:
                    if pq < 0:  # 第 2 步至少覆盖自己上一步那份
                        raise RuntimeError(
                            f"gin_overlap: 桶 {b} 的前任没找到（内部错误）"
                        )
                    d[b] = q - pq
        return off, d

    def _layout_host_windows(self):
        """两个组的 signal window / host window 布局（偏移和原来逐项相同，只是按组拆成两块、各自从 0 开始排）。
        返回 (dense 组 signal window 字节, 专家组 signal window 字节)，并填 recv_pool 的 credit / intra_arrive /
        gather / host_bytes。

        **dense 组（sharding 组）**：每个 dense 桶 16B signal（signal_offsets），之后
          credit[0] 机内（16 格，本机卡 l 打第 l 格）、credit[1] 机间（num_nodes 格），各自 256B 对齐；
          再接「机内到达」区 intra_arrive（只在 host window 里）：每个 dense 桶 16 格 u64，发送卡 l（本机卡号）写第 l 格
          （单写者，k_signal_intra_arrive），格偏移 = intra_arrive + (idx*16 + l)*8。固定 16 格（ranks_per_node ≤ 16），
          这样偏移不依赖它。取代原来每桶 [sigoff+8] 的共享计数（多卡对同一个 u64 atomicAdd_system，
          HOST_NATIVE_ATOMIC=0 下实测丢过加 1，所以改成每个发送方一格、单写者）；
          最后接 dense 2D 桶 gather 到 owner 的三块计数（按 dense 桶数留格，AdamW 下没人读写）：
            arrive + 8*idx           owner：跨节点发送方 put 带的 signal（网卡 atomic）
            intra  + (idx*16 + l)*8  owner：同节点发送方（组内本机卡 l）CE 拷完后写的序号（单写者 GPU store）
            credit + 8*idx           保留位（本路径不用 credit，没人写、没人读）
        **专家组（moe_sharding 组）**：每个专家桶 16B signal，之后 credit[2] 专家（专家组大小格，256B 对齐），
          再接专家 2D 桶 gather 的同样三块（按专家桶数留格，下标用专家桶号）。
        这些计数实际放在各组 host window 的同一偏移，signal window 里对应的这段只是占位、不再读写。
        所有 rank 按同样的算法算（算法不依赖 rank），gather_setup 里再 crc 校验。"""
        nd, nm = len(self.dense_bucket_numels), len(self.moe_bucket_numels)
        p = self.recv_pool
        # dense 组
        c0 = _align(_align(16 * nd + 64, 256), 256)
        c1 = _align(c0 + 8 * 16, 256)
        dense_signal_end = c1 + 8 * max(self.num_nodes, 1)
        ia = _align(dense_signal_end, 256)
        p["intra_arrive"] = ia
        p["intra_arrive_bytes"] = 8 * 16 * nd
        ga = _align(ia + p["intra_arrive_bytes"], 256)
        gi = _align(ga + 8 * max(nd, 1), 256)
        gc = _align(gi + 8 * 16 * max(nd, 1), 256)
        p["gather_dense"] = (ga, gi, gc)
        p["host_bytes"][GROUP_DENSE] = _align(gc + 8 * max(nd, 1), 4096)
        # 专家组
        c2 = _align(_align(16 * nm + 64, 256), 256)
        moe_signal_end = c2 + 8 * max(self.expert_group_size, 1)
        mga = _align(moe_signal_end, 256)
        mgi = _align(mga + 8 * max(nm, 1), 256)
        mgc = _align(mgi + 8 * 16 * max(nm, 1), 256)
        p["gather_moe"] = (mga, mgi, mgc)
        p["host_bytes"][GROUP_MOE] = _align(mgc + 8 * max(nm, 1), 4096)
        p["credit"] = (c0, c1, c2)
        return dense_signal_end, moe_signal_end

    def _configure(self, num_dense, num_moe):
        """侧流 / 每桶 event / 窗口 mark 环 + host window 里三块 credit 的字节偏移，一次交给 C++ 侧。
        桶数**只在这里**交给 C++ 侧一次（原来 configure 和 setup_buckets 各传一遍、release_setup 再传一遍，
        三处必须一致却没有核对）：放行层的 release_setup 不再带桶数，它从引擎按桶数开的那些 event 的长度读，
        传过去的固定放行顺序 / 每桶参数表的长度和它对不上就报错。"""
        c0, c1, c2 = self.recv_pool["credit"]
        check_cuda(
            device_engine().configure(
                num_dense, num_moe, NUM_WINDOW_MARKS, c0, c1, c2
            ),
            "configure",
        )
        self._log_recv_pool()

    # ------------------------------------------------------------------ 交给 GPU（放行层在 C++ 侧，见 release.cu）
    def _recv_offset(self, seg, idx):
        """桶 idx 在第 seg 段（0 机内 / 1 机间 = dense 组的 recv window；2 专家 = 专家组的 recv window）里的字节偏移
        = 段起点 + 环形映射表里的偏移（各 rank 一致，CE / put 写对端同一偏移）。"""
        p = self.recv_pool
        return p["base"][seg] + p["offsets"][seg].get(idx, 0)

    def release_setup(self):
        """放行层（调用方的放行时机层建起来时调一次，PaddleFleet 的 scheduler.__init__）：把它要的每桶参数
        一次传进 C++ 侧，并起尾部线程。
        每项都取自 release.cu 里 submit_* / blocked_* 要用的那些值（放行层不回调 Python）：
        dense 每桶 7 个 [send 偏移, signal 偏移, 本卡分片 numel, 机内段偏移, 机间段偏移, 机内前任距离, 机间前任距离]，
        专家每桶 5 个 [send 偏移, signal 偏移, 本卡分片 numel, 专家段偏移, 前任距离]（距离 None → -1 = 这一段没有槽）。
        桶数不传：放行层从引擎按桶数开的那些 event 的长度读（configure 是唯一交桶数的地方），这里传过去的
        固定顺序 / 每桶参数表长度和它对不上就报错。credit 三块偏移（configure 交过）、专家组大小 / 本卡组内序号
        （set_comms 交过）同理都不再传一遍。"""
        p, lib = self.recv_pool, device_engine()
        nd, nm = len(self.dense_bucket_numels), len(self.moe_bucket_numels)
        dtab = []
        for i in range(nd):
            so, sg = self.dense_offsets[i]
            di, dn = (
                p["distance"][0].get(i),
                p["distance"][1].get(i),
            )  # credit 目标里的前任距离，None → -1
            dtab += [so, sg, self.dense_bucket_numels[i] // self.group_size, self._recv_offset(0, i),
                     self._recv_offset(1, i), -1 if di is None else di, -1 if dn is None else dn]  # fmt: skip
        mtab = []
        for i in range(nm):
            so, sg = self.moe_offsets[i]
            dm = p["distance"][2].get(i)
            mtab += [so, sg, self.moe_bucket_numels[i] // self.expert_group_size, self._recv_offset(2, i),
                     -1 if dm is None else dm]  # fmt: skip
        check_cuda(
            lib.release_setup(
                p["dense_order"], p["moe_order"], dtab, mtab,
                PUT_BLOCKS, p["intra_arrive"], nd + nm + TAIL_EXTRA_MARKS,
            ),
            "release_setup",
        )  # fmt: skip

    # ------------------------------------------------------------------ Muon 2D 桶 gather 到 owner
    # 只有 optim=muon 走到这一段（PaddleFleet 的 backend._setup_muon 经它的 gather_setup 调到这里）；
    # AdamW 下没有 2D 桶，这几个方法一次都不会被调。
    def gather_table_setup(self, moe_2d, dense_2d, moe_group):
        """建 2D 桶表（集合操作：world 内所有 rank 同时调用；调用方在 bind_buckets 之后调）。
        moe_2d / dense_2d = 2D 桶对象（有 _gin_owner_bucket、_gin_bucket_id）。建每个 2D 桶的表 → crc 校验（dense 全 world、
        专家组内：桶号 / send 偏移 / 每段大小 / owner 一致，因为发送方按本卡的偏移写 owner 的 send window 和 host window）
        → C++ 侧 gather_setup。什么时候发这些 put（放行逻辑）在调用方那边（PaddleFleet 的
        scheduler.GatherToOwner）。
        表项（key = (kind, 桶号)）：group（GROUP_DENSE / GROUP_MOE，选 comm / window）、slot（C++ 侧 event 下标，
        dense 用桶号、专家用 dense 桶数 + 桶号）、off（本卡分片 seg(组内序号) 在本组 send window 的字节偏移）、m、
        组大小 S、本卡组内序号 me、owner 组内序号 o；owner 另有 n_inter（跨节点发送方数）、intra（同节点发送方在组内的
        本机序号）；host window 偏移 aoff / ioff / gi0（在本组的 host window 里）。
        其中 off / ioff / o 只用来建 C++ 侧那张表（tail_gather_setup），每次发 put 不再逐个传过去。"""
        import zlib

        p = self.recv_pool
        nd = len(self.dense_bucket_numels)
        tab = {}
        for kind, buckets in (("dense", dense_2d), ("moe", moe_2d)):
            ga, gi, _ = (
                p["gather_dense"] if kind == "dense" else p["gather_moe"]
            )
            L = (
                self.ranks_per_node
                if kind == "dense"
                else self.moe_ranks_per_node
            )
            ni = self.node_index if kind == "dense" else self.moe_node_index
            ml = self.rank_in_node if kind == "dense" else self.moe_rank_in_node
            for b in buckets:
                k, idx = b._gin_bucket_id
                assert k == kind, (k, kind)
                _, npad, o = b._gin_owner_bucket
                if kind == "dense":
                    grp, S, me, soff, slot = (
                        GROUP_DENSE,
                        self.group_size,
                        self.group_rank,
                        self.dense_offsets[idx][0],
                        idx,
                    )
                    assert npad == self.dense_bucket_numels[idx], (
                        idx,
                        npad,
                        self.dense_bucket_numels[idx],
                    )
                else:
                    grp, S, me = (
                        GROUP_MOE,
                        self.expert_group_size,
                        self.expert_group_rank,
                    )
                    soff, slot = self.moe_offsets[idx][0], nd + idx
                    assert npad == self.moe_bucket_numels[idx], (
                        idx,
                        npad,
                        self.moe_bucket_numels[idx],
                    )
                m = npad // S
                ent = {
                    "group": grp, "slot": slot, "send_offset": soff, "off": soff + me * m * 4, "m": m, "S": S,
                    "me": me, "o": o, "owner": o == me, "aoff": ga + 8 * idx,
                    "ioff": gi + (idx * 16 + ml) * 8, "gi0": gi + idx * 16 * 8,
                    "ranks_per_node": L, "node_index": ni,
                }  # fmt: skip
                if o == me:
                    peers = [j for j in range(S) if j != me]
                    ent["n_inter"] = sum(1 for j in peers if j // L != ni)
                    ent["intra"] = [j % L for j in peers if j // L == ni]
                else:
                    ent["intra_owner"] = o // L == ni
                tab[(kind, idx)] = ent

        def sig(kind):
            return [
                (i, e["send_offset"], e["m"], e["o"])
                for (k, i), e in sorted(tab.items())
                if k == kind
            ]

        def same(obj, group):
            h = zlib.crc32(repr(obj).encode()) & 0x7FFFFFFF
            v = paddle.to_tensor([h, -h], dtype="int64")
            dist.all_reduce(v, op=dist.ReduceOp.MAX, group=group)
            return int(v[0]) == -int(v[1])

        ok = same(
            (p["gather_dense"][:2], nd, self.ranks_per_node, sig("dense")),
            self.world,
        )
        if moe_group is not None and self.moe_bucket_numels:
            ok = same((p["gather_moe"][:2], len(self.moe_bucket_numels), self.moe_ranks_per_node, sig("moe")),
                      moe_group) and ok  # fmt: skip
        v = paddle.to_tensor([int(ok)], dtype="int32")
        dist.all_reduce(v, op=dist.ReduceOp.MIN, group=self.world)
        if int(v[0]) != 1:
            raise RuntimeError(
                "gin_overlap: 各 rank 的 2D 桶表（桶号 / send window 偏移 / 每段大小 / owner）或 host window 里 gather "
                "计数的偏移不一致（dense 要全 world 一致，专家要组内一致）；发送方按本卡的偏移写 owner，不一致会写错位置"
            )
        check_cuda(
            device_engine().gather_setup(nd + len(self.moe_bucket_numels)),
            "gather_setup",
        )
        self.gather_table = tab

        own = sum(1 for e in tab.values() if e["owner"])
        rx = sum(e["m"] * 4 * (e["S"] - 1) for e in tab.values() if e["owner"])
        log(
            f"init: 2D 桶 gather 到 owner：{len(tab)} 个桶（dense {len(dense_2d)} / 专家 {len(moe_2d)}），"
            f"本卡（rank 0）是 {own} 个的 owner、gather 收 {format(rx / 2**20, '.1f')} MiB；host window 计数 "
            f"dense arrive={p['gather_dense'][0]} intra={p['gather_dense'][1]}、专家 arrive={p['gather_moe'][0]} "
            f"intra={p['gather_moe'][1]}"
        )

    def tail_gather_setup(self):
        """init（gather 表建好之后；放行层已经起好、尾部线程在 IDLE）：把 2D 桶表交给放行层。
        每项 [kind(0 dense / 1 专家，同时决定用哪个组的 comm / window), 桶号, slot, 本卡是不是 owner,
        owner 组内序号, off, m, 1（保留位）, aoff, ioff]，取自 gather_table。
        这是这些 put 参数**唯一一次**过到 C++ 侧：之后主线程发 put 也只给表项下标（gather_put_entry），
        和尾部线程查的是同一张表。self.gather_keys = 表项顺序，gather_entry_index = (kind, 桶号) → 表项下标。"""
        keys = sorted(self.gather_table)
        self.gather_keys = keys
        self.gather_entry_index = {key: i for i, key in enumerate(keys)}
        if not keys:  # 没有 2D 桶：不建表
            return
        rows = []
        for kind, idx in keys:
            e = self.gather_table[(kind, idx)]
            rows += [int(kind == "moe"), idx, e["slot"], int(e["owner"]), e["o"], e["off"], e["m"], 1, e["aoff"],
                     e["ioff"]]  # fmt: skip
        check_cuda(device_engine().tail_gather_setup(rows), "tail_gather_setup")

    def tail_gather_set(self, seqs):
        """交接前（尾部线程在 IDLE）：本步每个表项的 put 序号（0 = 本步不归它发）。表为空时不调。"""
        if not self.gather_keys:
            return
        check_cuda(
            device_engine().tail_gather_set_sequences(list(seqs)),
            "tail_gather_set_sequences",
        )

    def tail_gather_collect(self):
        """尾部线程已离开 RUN：本步它发的 gather put，[((kind, 桶号), 序号)]，按发出顺序。"""
        if not self.gather_keys:
            return []
        rc, flat = device_engine().tail_gather_get_sent()
        check_cuda(rc, "tail_gather_get_sent")
        return [
            (
                ("moe" if flat[3 * i] else "dense", int(flat[3 * i + 1])),
                int(flat[3 * i + 2]),
            )
            for i in range(len(flat) // 3)
        ]

    # ------------------------------------------------------------------ 初始化日志
    def _log_recv_pool(self):
        """接收池的 init 日志（rank 0）：三段大小、下限、本卡最大桶、按本段桶槽中位数能同时放几个桶，以及前任在本步内的桶数。"""
        p = self.recv_pool
        names = ("dense 机内段", "dense 机间段", "专家段")
        segs = []
        for i in range(3):
            (bi, bsz), (med, k) = p["big"][i], p["fit"][i]
            segs.append(
                f"{names[i]} {_gib(p['P'][i])} GiB（下限 {_gib(p['lb'][i])}，本卡最大桶 "
                f"{'dense' if i < 2 else '专家'}#{bi} 槽 {_gib(bsz)} GiB，槽中位数 {format(med / 2**20, '.1f')} MiB "
                f"可同时放 {k} 个）"
            )
        nd, nm = len(self.dense_bucket_numels), len(self.moe_bucket_numels)
        n = (nd, nd, nm)
        inner = [
            sum(1 for v in p["distance"][i].values() if v < n[i])
            for i in range(3)
        ]  # d < n：要等同一步前面的桶腾空
        hw = p["host_window_bytes"]
        hw_txt = "、".join(
            f"{GROUP_NAME[g]} 组 {format(hw[g] / 2**20, '.1f')} MiB（需要 {p['host_bytes'][g]} B）"
            for g in sorted(hw)
        )
        log(
            "init: 接收池（环形映射、多桶同时在途，CPU 直接读 host window 判断后再交）"
            + "；".join(segs)
            + f"；共 {_gib(p['total'])} GiB（机内段 / 机间段在 sharding 组的 recv window，专家段在 moe_sharding 组的）；"
            f"前任在本步内的桶 机内 {inner[0]} / 机间 {inner[1]} / 专家 {inner[2]} 个；"
            f"host window 每卡 {hw_txt}（HOST_NUMA；机内到达区在 sharding 组那块、偏移 {p['intra_arrive']}、"
            f"{p['intra_arrive_bytes']} B = {nd} 桶 × 16 格 × 8 B，每发送方一格）"
        )

    def _log_init(self, t, extra=""):
        """每个 rank 一行初始化各阶段耗时（千卡上建连可能明显变慢）；rank 0 再打拓扑 / 桶 / 流量 / 环境。"""
        ms = [format((b - a) * 1e3, ".0f") for a, b in zip(t, t[1:])]
        print(
            f"[gin_overlap init] rank {dist.get_rank()} 节点 {self.node_index} 卡 {self.rank_in_node}："
            f"取 Paddle 的两个 ncclComm {ms[0]} / window 注册 + DevCommCreate {ms[1]} / setup {ms[2]} / "
            f"收尾 barrier {ms[3]} ms",
            flush=True,
        )
        if dist.get_rank() != 0:
            return
        L, Nn, N, W = (
            self.ranks_per_node,
            self.num_nodes,
            self.expert_group_size,
            self.group_size,
        )
        v = device_engine().nccl_runtime_version()
        p = self.recv_pool
        moe_blk = [n // N * 4 / 2**20 for n in self.moe_bucket_numels] or [0]
        den_blk = [n // W * 4 / 2**20 for n in self.dense_bucket_numels] or [0]
        net = sum((N - 1) * (n // N) * 4 for n in self.moe_bucket_numels)
        net += sum((Nn - 1) * (n // W) * 4 for n in self.dense_bucket_numels)
        ce = sum((L - 1) * Nn * (n // W) * 4 for n in self.dense_bucket_numels)
        nput = len(self.moe_bucket_numels) * (N - 1) + len(
            self.dense_bucket_numels
        ) * (Nn - 1)
        envs = " ".join(
            f"{k}={os.getenv(k, '<unset>')}"
            for k in (
                "NCCL_IB_TC",
                "NCCL_GIN_IB_TC",
                "NVSHMEM_IB_TRAFFIC_CLASS",
                "NCCL_NVLS_ENABLE",
                "CUDA_DEVICE_MAX_CONNECTIONS",
                "CUDA_MODULE_LOADING",
            )
        )
        wins = "；".join(
            f"{GROUP_NAME[g]} 组 send {_gib(self._win[g][0])} + recv {_gib(self._win[g][1])} GiB（GIN ctx="
            f"{p['nctx'][g]}）" for g in sorted(self._win)
        )  # fmt: skip
        log(
            f"init: NCCL {v // 10000}.{v // 100 % 100}.{v % 100}，sharding 组 {Nn} 节点 × {L} 卡（组大小 {W}，每机卡数 "
            f"{self._local_size}）、moe_sharding 组 {self.moe_num_nodes} 节点 × {self.moe_ranks_per_node} 卡"
            f"（组大小 {N}）；window {wins}"
        )
        log(
            f"init: 专家 {len(self.moe_bucket_numels)} 桶（组 {N} rank，peer 用组内序号，每 peer 一笔 put，"
            f"每 peer 块 {format(min(moe_blk), '.3f')}~{format(max(moe_blk), '.3f')} MB）；"
            f"dense {len(self.dense_bucket_numels)} 桶（机外 {Nn - 1} 个同卡号 peer，每 peer 块 "
            f"{format(min(den_blk), '.3f')}~{format(max(den_blk), '.3f')} MB）"
        )
        log(
            f"init: 每步每卡机间 {format(net / 1e9, '.2f')} GB / {nput} 次 put，机内 CE {format(ce / 1e9, '.2f')} GB"
        )
        # 调用方自己那点东西（PaddleFleet 的 gin_overlap 配置）跟在环境变量前面，
        # 原来这一行就是 log(f"init: {cfg}；{envs}")，位置和内容都没变
        log(f"init: {extra}{envs}")
