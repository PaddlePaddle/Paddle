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

"""gin_overlap 的通信原语：用 NCCL GIN 单边写分阶段完成 sharding stage1 每个梯度桶的 reduce_scatter。

这个文件负责：window / 接收池 / host window 的布局，两个通信组的建立和校验，交给 C++ 侧放行层的每桶参数表，
2D 桶 gather 的表，以及初始化日志。这里是 Python 侧的完整说明，其它文件只引用这里。
主要接口：device_engine()（C++ 引擎对象）、GinOverlapPrimitive（调用方继承它）、check_cuda、log。
什么时候推进哪个阶段（放行时机）不在这一层，由调用方决定。

术语
  - 桶（bucket）：一块连续显存，装着若干参数的梯度；通信以桶为单位。
  - 分片（shard）：桶按通信组切开后，本卡负责的那一片。
  - GIN：NCCL 的 GPU-Initiated Networking。这里只用它的单边写 put：把本卡一段数据写进对端的 window，
    对端 GPU 不参与。
  - LSA：NCCL 设备 API 里「同一节点、能用指针直接访问」的那组卡；lsaRank / lsaSize = 节点内序号 / 节点内卡数。
  - devcomm：ncclDevComm，kernel 里调用 GIN / LSA 用的通信器句柄。
  - CE：拷贝引擎（copy engine），机内拷贝走它，不占 SM。
  - window：注册给 NCCL 的对称内存，对端能单边读写。每个通信组一套 send / recv / signal 显存 window，
    外加一块 host window（CPU 内存，放计数）。各桶的 grad_storage 就是本组 send window 的一段。
  - 窗口（时间上的）：反向里 DeepEP 不占网卡或不占 SM 的一段时间。
  - mark：计算流上记的 cudaEvent，提交到 GPU 的工作先等它。
  - 放行：判断某个桶的某个阶段现在可以做，并把它提交到 GPU（C++ 侧放行层，release.cu）。
  - credit：接收方用完接收池里的一个槽后，给发送方的计数 +1（发 credit）；发送方看到计数达标才能写那个槽。
  - owner：Muon 2D 桶最终归哪张卡，那张卡就是它的 owner。
  - 组内序号：通信组内部的 rank 编号，不是 world rank。
  - 尾部线程：C++ 放行层的后台线程，本步最后一个窗口之后由它放行。
  - 分片求和：dense 桶把各节点的部分和加起来、专家桶把专家组各卡的那一份加起来，得到本卡分片的最终结果。
    代码里叫 inter_node_reduce（dense）/ moe_reduce（专家），两者合称 shard_reduce。

通信组
  - dense 桶走 sharding 组（GROUP_DENSE），专家桶走 moe_sharding 组（GROUP_MOE），各自一套 comm / devcomm /
    window。comm 取自 Paddle 已有的 ProcessGroup。
  - 两个组都必须 rank 列表升序、每个节点成员数相同：rank_in_node = 组内序号 % ranks_per_node，
    node_index = 组内序号 // ranks_per_node。

阶段
  - dense 桶：机内拷贝（CE）→ 机内求和 → 同卡号机间 put → 分片求和。
  - 专家桶：put 给专家组其余各卡 → 分片求和。
  - Muon 2D 桶：分片求和之后，把本卡分片 gather 到 owner（跨节点 GIN put，同节点 CE）。

接收池（recv window，_layout_recv_pool）
  - 三段：dense 机内段、dense 机间段（都在 dense 组的 recv window），专家段（专家组的 recv window）。
  - 每段是一个环：桶按固定放行顺序依次占一个槽，放不下就绕回段首，所以多个桶可以同时在途。
  - 前任距离 d：本桶的槽上一个占用者在放行序号上比本桶小多少（1..n，n = 前任是本桶自己上一步那份）。
    放行层按它算 credit 目标。
  - 布局必须所有 rank 一致（dense 两段全 world 一致，专家段组内一致），初始化时 crc 校验。

host window 布局（_layout_host_windows，每格一个 u64）
  - dense 组：
    - signal：每个 dense 桶 16 B，第一格是机间 put 带的 GIN signal。
    - credit[0] 机内：16 格，同节点卡 l 发来的 credit 记在第 l 格。
    - credit[1] 机间：num_nodes 格，节点 n 同卡号的卡发来的 credit 记在第 n 格。
    - intra_arrive 机内到达格：每个 dense 桶 16 格，格偏移 = intra_arrive + (桶号*16 + l)*8，同节点发送卡 l
      写入本桶已提交的机内拷贝次数。
    - gather_dense：dense 2D 桶 gather 的 (arrive, intra, credit) 三块，见 _layout_host_windows。
  - 专家组：每个专家桶 16 B signal；credit[2] 专家，专家组大小格；gather_moe 三块。

约束
  - 计数跨步累加、只增不减；CPU 不能写这些格子，也不能清零（_hw 只读）。
  - 每张发送卡只写自己那一个机内到达格，同一格不能有多个写者。
  - 每个阶段必须按固定放行顺序提交到 GPU，所有 rank 的顺序必须相同。
  - 每个桶每步都必须走完全部阶段，否则后面的 credit 永远等不到。
"""

import ctypes
import os
import time
from collections import Counter

import paddle
import paddle.distributed as dist

# C++ 实现（paddle.base.core.gin_overlap，源码在 paddle/fluid/distributed/collective/gin_overlap/）
# 只在满足构建条件时才有，否则 paddle.base.core 上没有 gin_overlap 属性。下面是报错时给出的条件说明。
_BUILD_HINT = (
    "这个 paddlepaddle 没有编 gin_overlap：要 WITH_GPU + WITH_NCCL、NCCL >= 2.30.7 "
    "（带 nccl_device.h）、CUDA arch 含 9.0 或 10.0"
)

# 固定参数（都不影响数值结果）
PUT_BLOCKS = 64  # put kernel 的 block 数，由 release_setup 传给放行层
GIN_CONTEXTS = 8  # 请求的 GIN context 数（实际数以 devcomm 返回为准）
TIMEOUT_S = 300.0  # 等对端的超时（秒），超时报错
NUM_WINDOW_MARKS = 1024  # 窗口 mark 环的大小
# 尾部 mark 环的大小 = 桶数 + TAIL_EXTRA_MARKS。每步要用 1（交接）+ 交接后 ready 的桶数 + 交接后经过的 combine 数
#   个槽；一步内用完时报错，不能绕回
TAIL_EXTRA_MARKS = 1024
LOG_EVERY = 100  # 每多少步输出一行累计流量 / 调度汇总日志
# 通信组编号，必须和 C++ 侧的 GROUP_DENSE / GROUP_MOE 一致（device_engine() 里核对）
GROUP_DENSE, GROUP_MOE = 0, 1
GROUP_NAME = {GROUP_DENSE: "sharding", GROUP_MOE: "moe_sharding"}

_DEVICE_ENGINE = None


def device_engine():
    """返回进程内唯一的 paddle.base.core.gin_overlap.GradCommEngine 对象，第一次调用时创建。
    没有 paddle.base.core.gin_overlap、通信组编号和 C++ 侧不一致、或编译期和运行时的 NCCL 版本不一致时报错。"""
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
    """每台机器的卡数：PADDLE_LOCAL_SIZE；没设或不是正整数时用可见设备数。"""
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
        """建两个通信组的 window、devcomm 和接收池布局，并交给 C++ 侧（configure）。
        集合操作：world（sharding 组）内所有 rank 必须同时调用；有专家桶时专家组（moe_sharding）内所有 rank 也必须同时调用。
          - world / moe_group：sharding 组 / 专家组对象（没有专家桶时 moe_group 可以是 None）。
          - moe_numels / dense_numels：各桶的元素数，必须能被各自的组大小整除。
          - release_order：(dense 桶固定放行顺序, 专家桶固定放行顺序)，都是桶号的排列，所有 rank 必须一致。
          - dense_recv_pool_gb / moe_recv_pool_gb：dense 接收池（机内段 + 机间段）和专家段的大小（GiB）。
        布局见模块 docstring；之后调用方还要调 release_setup。"""
        device_engine()  # 尽早创建引擎对象并做版本核对
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

        # 组内节点分块（要求见模块 docstring）；之后在 _check_devcomm 里和 devcomm 的 lsaRank / lsaSize 核对
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

        # 每组自己的 send window，桶按字节偏移排（两组各自从 0 开始）
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
        # 每桶 16 B：[0] 机间 / 专家 GIN signal，[1] 未使用
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
        # 各 rank 按自己的大小注册 window；对端访问不能越界
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
        """让 C++ 侧按 ring id（通信组对象的 `.id`）取两个组在 Paddle 里已有的 ncclComm。
        C++ 侧会核对组大小和组内序号；取不到或对不上时报错。组还没有 NCCL comm 时会在这里创建。"""
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
        """建一个组的全部 window（集合操作，组内所有 rank 都要调）：alloc_windows → alloc_host_counters（必须在
        create_dev_comm 之前）→ create_dev_comm → _check_devcomm → set_group_topology。"""
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
        if nbytes < host_bytes:  # C++ 侧按粒度向上取整，正常不会小于
            raise RuntimeError(
                f"gin_overlap: {what} 组的 host window {nbytes} B 小于需要的 {host_bytes} B"
            )
        self.recv_pool["host_window_bytes"][gid] = nbytes
        # host window 的 u64 视图（ctypes 数组，每次下标访问都读一次内存）。只能读，不能写
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
        """返回 (每节点组内成员数 ranks_per_node, 每机卡数 local_size)。组的 rank 列表必须升序、每个节点的成员数
        必须相同（可以小于每机卡数），否则报错。"""
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
        """核对 devcomm 的 lsaRank / lsaSize 和 _topology 推算的一致（集合操作；组内任一 rank 不满足，全组报错）。"""
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
        """接收池三段的布局（集合操作）：算每段大小、每个桶在段里的偏移和前任距离，并校验各 rank 一致。
          - 每段的下限 = 本段最大桶的槽大小（256 B 对齐后向上取整到 4 KB）；dense 两段在全 world 取最大，专家段在
            专家组内取最大。
          - dense_recv_pool_gb 按两段下限的比例分给机内段和机间段（各自向下对齐到 4 KB，必须 ≥ 下限）；
            moe_recv_pool_gb 是专家段（向下对齐到 4 KB）。
          - 机内段、机间段依次放在 dense 组的 recv window，专家段放在专家组的 recv window，都从 0 开始。
        结果存进 self.recv_pool，返回 (dense recv window 字节数, 专家 recv window 字节数)。"""
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

        # 发送方按本卡的表算对端的写入偏移，所以表必须所有 rank 一致
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
        ):  # 按槽大小中位数估算这一段能同时放几个桶（只用于日志）
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
            "distance": (d_i, d_n, d_m),  # 每段 {桶号: 前任距离 d}（1..n；n = 前任是本桶上一步那份）
            "dense_order": d_order, "moe_order": m_order,  # 固定放行顺序（放行层的「位置」是这里的下标）
            "buckets": None,  # 调用方填 (专家桶, dense 桶) 的桶对象
            "host_bytes": [0, 0],  # _layout_host_windows 填：两组 host window 各需要多少字节
            "host_window_bytes": {},  # _setup_group 填：实际大小（按 HOST_NUMA 粒度取整）
            "nctx": {},  # _setup_group 填：两组各自的 GIN ctx 数
        }  # fmt: skip
        return P[0] + P[1], P[2]

    def _recv_pool_sizes(self, lb, descs):
        """三段的实际大小（字节，4 KB 的倍数）。lb = 三段下限，descs = 报错用的下限说明。"""
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
                if P[i] < lb[i]:  # lb 是 4 KB 的倍数且 Td ≥ Sd 时不会发生
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
        """一段的环形映射：桶按 order 依次占一块连续空间，放不下（cur + size > P）就绕回 0，段尾留空。每步布局相同。
        前任 = 本桶的槽覆盖的区间里，之前的占用者中序号最大的那个（序号 = 步 × n + 在 order 里的位置）。
        前任距离 d = 本桶序号 - 前任序号，取第 2 步的结果（1..n；n = 前任是本桶上一步那份）。
        返回 ({桶: 偏移}, {桶: d})；槽大小为 0 的桶不在表里（这一段不用，也不等 credit）。"""
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
        """两个组的 signal window / host window 布局，各自从 0 开始。返回 (dense 组 signal window 字节,
        专家组 signal window 字节)，并填 recv_pool 的 credit / intra_arrive / gather_* / host_bytes。

        dense 组（sharding 组），各块 256 B 对齐，依次是：
          - signal：每个 dense 桶 16 B（signal_offsets），第一格是机间 GIN signal。
          - credit[0] 机内：16 格，同节点卡 l 发来的 credit 记在第 l 格。
          - credit[1] 机间：num_nodes 格。
          - intra_arrive 机内到达格（只在 host window 里有意义）：每个 dense 桶固定 16 格（ranks_per_node 必须 ≤ 16），
            格偏移 = intra_arrive + (idx*16 + l)*8，只有同节点发送卡 l 写第 l 格，写的是本桶已提交的机内拷贝次数。
          - gather_dense = (arrive, intra, credit)，dense 2D 桶 gather 到 owner 用，按 dense 桶数留格：
            - arrive + 8*idx：owner 上的跨节点到达计数，每个跨节点发送方的 put 让它 +1。
            - intra + (idx*16 + l)*8：owner 上的同节点到达格，同节点发送方 l 在 CE 之后写入序号。
            - credit + 8*idx：保留位，不读不写。
        专家组（moe_sharding 组）：每个专家桶 16 B signal；credit[2] 专家（专家组大小格）；gather_moe 三块，
          格式同 gather_dense，按专家桶号留格。
        计数实际写在 host window 的同一偏移，signal window 只占位。布局只取决于桶数和组大小，所有 rank 相同。"""
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
        """调 C++ 侧 configure：传入桶数、窗口 mark 环大小和三块 credit 区的偏移，然后输出接收池的初始化日志。
        桶数只在这里传给 C++ 侧。"""
        c0, c1, c2 = self.recv_pool["credit"]
        check_cuda(
            device_engine().configure(
                num_dense, num_moe, NUM_WINDOW_MARKS, c0, c1, c2
            ),
            "configure",
        )
        self._log_recv_pool()

    # ------------------------------------------------------------------ 放行层的参数表（放行层在 C++ 侧 release.cu）
    def _recv_offset(self, seg, idx):
        """桶 idx 在第 seg 段的 recv window 字节偏移（0 机内 / 1 机间：dense 组的 recv window；2 专家：专家组的）
        = 段起点 + 段内偏移。"""
        p = self.recv_pool
        return p["base"][seg] + p["offsets"][seg].get(idx, 0)

    def release_setup(self):
        """把固定放行顺序和每桶参数表交给 C++ 放行层，并启动尾部线程。在 __init__ 之后调一次。
        - dense 每桶 7 个数：[send 偏移, signal 偏移, 本卡分片 numel, 机内段偏移, 机间段偏移, 机内前任距离,
          机间前任距离]。
        - 专家每桶 5 个数：[send 偏移, signal 偏移, 本卡分片 numel, 专家段偏移, 前任距离]。
        - 前任距离为 -1 表示这一段没有槽。"""
        p, lib = self.recv_pool, device_engine()
        nd, nm = len(self.dense_bucket_numels), len(self.moe_bucket_numels)
        dtab = []
        for i in range(nd):
            so, sg = self.dense_offsets[i]
            di, dn = (
                p["distance"][0].get(i),
                p["distance"][1].get(i),
            )  # 前任距离，None → -1
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
    # 只有 Muon（有 2D 桶）才用到这一段。
    def gather_table_setup(self, moe_2d, dense_2d, moe_group):
        """建 2D 桶表，校验各 rank 一致，再调 C++ 侧 gather_setup。集合操作：world 内所有 rank 必须同时调用。
        moe_2d / dense_2d = 2D 桶对象（必须有 _gin_owner_bucket、_gin_bucket_id）。dense 表在全 world、专家表在专家组内
        必须一致（桶号 / send 偏移 / 每段大小 / owner），否则报错。
        结果存进 self.gather_table，key = (kind, 桶号)，每项：
          - group：GROUP_DENSE / GROUP_MOE；slot：C++ 侧 event 下标（dense 用桶号，专家用 dense 桶数 + 桶号）。
          - send_offset：桶在本组 send window 的偏移；off：本卡分片的偏移；m：每片元素数；S：组大小；
            me：本卡组内序号；o：owner 组内序号；owner：本卡是不是 owner。
          - aoff / ioff / gi0：本组 host window 里的跨节点到达格 / 本卡的同节点到达格 / 本桶同节点到达格的起点。
          - owner 另有 n_inter（跨节点发送方数）、intra（同节点发送方的节点内序号列表）；发送方另有 intra_owner
            （owner 是否在同一节点）。"""
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
        """把 2D 桶表交给 C++ 放行层（release_setup 和 gather_table_setup 之后调，尾部线程必须在 IDLE）。
        每项 [kind（0 dense / 1 专家）, 桶号, slot, 本卡是不是 owner, owner 组内序号, off, m, 1（保留位）, aoff, ioff]。
        之后发 gather put 只传表项下标。self.gather_keys = 表项顺序；self.gather_entry_index = (kind, 桶号) → 表项下标。"""
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
        """把本步每个表项的 put 序号（0 = 本步不由尾部线程发）交给 C++ 侧。必须在交给尾部线程之前调。"""
        if not self.gather_keys:
            return
        check_cuda(
            device_engine().tail_gather_set_sequences(list(seqs)),
            "tail_gather_set_sequences",
        )

    def tail_gather_collect(self):
        """本步尾部线程发出的 gather put，[((kind, 桶号), 序号)]，按发出顺序。尾部线程必须已离开 RUN。"""
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
        """接收池的初始化日志（rank 0）：三段大小、下限、本卡最大桶、按槽大小中位数估算能同时放几个桶，以及前任在本步内的桶数。"""
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
        ]  # d < n：要等同一步里前面的桶空出槽
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
        """每个 rank 输出一行初始化各阶段耗时；rank 0 再输出拓扑、桶、流量和环境变量。extra 插在环境变量前面。"""
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
        log(f"init: {extra}{envs}")
