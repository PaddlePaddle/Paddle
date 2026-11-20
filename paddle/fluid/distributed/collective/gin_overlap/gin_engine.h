// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// gin_engine.h —— gin_overlap 的总声明：全局状态（window、侧流、event、
// 偏移）、错误检查宏、各阶段的提交函数、放行层接口，以及导出给 Python 的
// GradCommEngine 类。
//
// gin_overlap 把 sharding stage1 每个梯度桶的 reduce_scatter 拆成几个
// 阶段，用 NCCL 的 GIN 单边写分步完成，每个阶段在反向里单独提交到 GPU。
// 这里是 C++ 侧的完整说明，其它文件只引用这里。
//
// 术语
//   - 桶（bucket）：一块连续显存，装着若干参数的梯度；通信以桶为单位。
//   - 分片（shard）：桶按通信组切开后，本卡负责的那一片。
//   - GIN：NCCL 的 GPU-Initiated Networking。这里只用它的单边写 put：
//     把本卡一段数据写进对端的 window，对端 GPU 不参与。
//   - LSA：NCCL 设备 API 里「同一节点、能用指针直接访问」的那组卡
//     （Load/Store Accessible）。lsaRank / lsaSize = 本卡在节点内的
//     序号 / 节点内的卡数。
//   - devcomm：ncclDevComm，kernel 里调用 GIN / LSA 用的通信器句柄。
//   - CE：拷贝引擎（copy engine）。cudaMemcpyBatchAsync 走它，不占 SM。
//   - window：注册给 NCCL 的对称内存，对端能单边读写。每个通信组一套
//     send / recv / signal 三块显存 window，外加一块 host window（CPU
//     内存，放计数）。signal window 只占位，计数实际写在 host window 的
//     同一偏移上。send window 里各桶那一段就是该桶的 grad_storage
//     （send_window_tensor 包成张量，零拷贝）。
//   - 窗口（时间上的）：反向里 DeepEP 不占网卡或不占 SM 的一段时间。
//   - mark：计算流上记的 cudaEvent。提交到 GPU 的工作先等 mark，所以
//     不会早于这个时刻开始，梯度也已经写完。
//   - 放行：判断某个桶的某个阶段现在可以做，并把它提交到 GPU。
//   - credit：接收方用完接收池里的一个槽后，给发送方的计数 +1，叫「发
//     credit」。发送方看到计数达标，才能往那个槽里写。
//   - owner：Muon 2D 桶最终归哪张卡，那张卡就是这个桶的 owner。
//   - 组内序号：通信组内部的 rank 编号，不是 world rank。
//   - 尾部线程：放行层的后台线程，本步最后一个窗口之后由它放行。
//
// 通信组
//   - dense 桶走 sharding 组（GROUP_DENSE），专家桶走 moe_sharding 组
//     （GROUP_MOE）。两组各有自己的 ncclComm、devcomm 和 window。
//   - 组内必须按节点分块：rank_in_node = rank % ranks_per_node，
//     node_index = rank / ranks_per_node。
//
// dense 桶的四个阶段。卡 (node_index, rank_in_node) 负责第 rank_in_node
// 列，即块 {n * ranks_per_node + rank_in_node | n = 0..num_nodes-1}：
//   1. 机内拷贝：CE 把本卡在第 l 列的各块推进同节点卡 l 的 recv window
//      机内段里本卡那一行，再写卡 l 的 host window 里的机内到达格。
//   2. 机内求和：本列各块 = 本节点各卡对应块之和，原地写回 grad，作为
//      put 的源；之后给同节点其余卡发机内 credit。
//   3. 机间 put：本列第 n 块发给节点 n 上同卡号的卡，写进它机间段的第
//      node_index 行，同时它 host window 里本桶的 signal 计数 +1。
//   4. 分片求和：本卡分片 = 各节点部分和之和 × scale；之后给其余节点
//      同卡号的卡发机间 credit。
// 专家桶两个阶段：put 给专家组其余各卡，再做分片求和、发专家 credit。
// Muon 2D 桶：分片求和之后，再把本卡分片 gather 到 owner（跨节点用 GIN
// put，同节点用 CE + release store）。
//
// 「分片求和」在代码里叫 inter_node_reduce（dense）、moe_reduce（专家），
// 两者合称 shard_reduce（如 g_stream_shard_reduce、ADVANCE_SHARD_REDUCE）。
//
// 接收池（recv window）
//   - dense 组的 recv window = [机内段][机间段]，专家组的 = [专家段]。
//   - 每段是一个环：桶按固定放行顺序依次占一个槽，放不下就绕回段首，
//     所以同一段里可以有多个桶同时在途。
//   - 一个槽被新桶复用之前，接收方必须已经用完占着它的那个桶，并发了
//     credit。
//
// host window 布局（每组一块 HOST_NUMA 内存，每格一个 u64，偏移由
// Python 侧算好传进来）
//   - dense 组：
//     - [signal_offset]：每个 dense 桶一格，机间 put 带的 GIN signal。
//     - [intra_arrive + (bucket*16 + l)*8]：机内到达格，同节点发送卡 l
//       写入本桶已提交的机内拷贝次数。
//     - [credit_intra + 8*l]：同节点卡 l 做完一次机内求和后 +1。
//     - [credit_inter + 8*n]：节点 n 同卡号的卡做完一次分片求和后 +1。
//     - 2D 桶 gather 的到达格。
//   - 专家组：
//     - [signal_offset]：每个专家桶一格，专家 put 带的 GIN signal。
//     - [credit_moe + 8*k]：组内序号 k 做完一次分片求和后 +1。
//     - 2D 桶 gather 的到达格。
//
// 约束
//   - 计数跨步累加、只增不减；CPU 不能写这些格子，也不能清零。
//   - 同一个 u64 不能既有网卡原子写又有 GPU 写：signal 格和机间 / 专家
//     credit 只由网卡写，机内到达格和机内 credit 只由 GPU 写。
//   - 每张发送卡只写自己那一个机内到达格，同一格不能有多个写者。
//   - 每个阶段必须按固定放行顺序提交到 GPU，所有 rank 的顺序必须相同；
//     credit 的目标值是按这个顺序算的。
//   - 提交到 GPU 的工作只等本卡的 event（计算流上的 mark、本卡侧流的
//     event），不能等远端计数。远端计数由放行层（release.cu）在 CPU 上
//     读 host window 判断。
//   - 每个桶每步都必须走完全部阶段，否则后面的 credit 永远等不到。
//   - 放行状态任何时刻只能有一个写者：主线程（advance）或尾部线程。
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <nccl_device.h>

#include "paddle/fluid/distributed/collective/gin_overlap/gin_dynload.h"
#include "paddle/fluid/distributed/collective/gin_overlap/kernels/kernels.h"

namespace gin_overlap {

// 返回码：0 成功；<10000 = cudaError_t；>=10000 = 10000 + ncclResult_t
#define CUDA_CHECK(x)                          \
  do {                                         \
    cudaError_t _err = (x);                    \
    if (_err != cudaSuccess) return (int)_err; \
  } while (0)
#define NCCL_CHECK(x)                                  \
  do {                                                 \
    ncclResult_t _res = (x);                           \
    if (_res != ncclSuccess) return 10000 + (int)_res; \
  } while (0)
// 驱动 API 出错：打印 CUresult，再返回 cudaErrorUnknown（CUresult 和
// cudaError_t 的编号不通用）
#define CUDA_DRIVER_CHECK(x)                    \
  do {                                          \
    CUresult _res = (x);                        \
    if (_res != CUDA_SUCCESS) {                 \
      const char* _msg = nullptr;               \
      cuGetErrorString(_res, &_msg);            \
      fprintf(stderr,                           \
              "[gin_overlap] %s 失败: %d %s\n", \
              #x,                               \
              (int)_res,                        \
              _msg ? _msg : "?");               \
      return (int)cudaErrorUnknown;             \
    }                                           \
  } while (0)

// ----------------------------------------------------------------
// 通信组：每组一套 ncclComm、devcomm 和 window（分组方式见文件头）。
// 组内寻址一律用组内序号。
enum { GROUP_DENSE = 0, GROUP_MOE = 1, NUM_GROUPS = 2 };

struct CommGroup {
  ncclComm_t comm = nullptr;
  ncclDevComm_t dev_comm{};
  int num_gin_contexts =
      0;  // devcomm 实际给的 GIN context 数（put kernel 按 blockIdx 轮转）
  int size = 0, rank = 0;  // 组大小、本卡组内序号
  int ranks_per_node = 1;  // 一个节点上的组内成员数（= dev_comm.lsaSize）
  int rank_in_node = 0;  // 本卡在节点内的序号（= dev_comm.lsaRank）
  int num_nodes = 1;     // 组内节点数
  int node_index = 0;    // 本卡所在节点的序号
  ncclWindow_t win_send{}, win_recv{}, win_signal{}, win_host{};
  char* send = nullptr;  // send window：本组各桶的 grad_storage
  char* recv = nullptr;  // dense 组 = [机内段][机间段]；专家组 = [专家段]
  char* signal = nullptr;  // 只占位：计数写在 host window 的同一偏移
  char* host = nullptr;  // host window 的 CPU 地址（放行层从这里读计数）
  size_t send_bytes = 0, host_bytes = 0;
  CUmemGenericAllocationHandle host_handle = 0;
  char* lsa_recv[16] =
      {};  // 同节点卡 l 的 recv / signal / host / send window 起点
  char* lsa_signal[16] = {};
  char* lsa_host[16] = {};
  char* lsa_send[16] = {};
  int dev_comm_ready = 0;
};
extern CommGroup g_groups[NUM_GROUPS];
inline CommGroup& group_at(int group) { return g_groups[group]; }
inline CommGroup& dense_group() { return g_groups[GROUP_DENSE]; }
inline CommGroup& expert_group() { return g_groups[GROUP_MOE]; }
inline bool is_valid_group(int group) {
  return group == GROUP_DENSE || group == GROUP_MOE;
}
inline const char* group_name(int group) {
  return group == GROUP_DENSE ? "sharding"
                              : (group == GROUP_MOE ? "moe_sharding" : "?");
}

// ----------------------------------------------------------------
// 全局状态（gin_engine.cc 里定义）
extern int g_device_id;  // 本进程用的卡（set_comms 里设）
extern cudaStream_t g_stream_intra_node_copy;    // 机内拷贝
extern cudaStream_t g_stream_intra_node_reduce;  // 机内求和
extern cudaStream_t g_stream_put;                // dense put + 专家 put
extern cudaStream_t g_stream_shard_reduce;       // dense + 专家分片求和
extern std::vector<cudaEvent_t> g_event_intra_node_copy,
    g_event_intra_node_reduce, g_event_dense_put, g_event_moe_put;
extern std::vector<cudaEvent_t> g_event_dense_inter_node_reduce,
    g_event_moe_reduce;
extern std::vector<cudaEvent_t> g_window_marks;  // 窗口 mark 环（循环复用）
// credit 区在 host window 里的字节偏移：机内 / 机间在 dense 组的 host
// window，专家在专家组的 host window
extern size_t g_credit_intra_offset, g_credit_inter_offset, g_credit_moe_offset;
extern int g_buckets_ready;

// ----------------------------------------------------------------
// Paddle 当前计算流（实现在 paddle_tensor.cc），和 Python 的
// paddle.device.current_stream() 是同一条流。取不到时打印原因并返回错误码，
// 不退回默认流。
int paddle_compute_stream(cudaStream_t* out);

// ----------------------------------------------------------------
// 按 ring id（Python 侧通信组对象的 .id）取 Paddle ProcessGroup 已有的
// ncclComm_t，同时返回组大小和本卡序号（实现在 paddle_comm.cc）。
// 返回非 0 时 *out_err 是原因。
int paddle_nccl_comm(int ring_id,
                     int device_id,
                     ncclComm_t* out_comm,
                     int* out_size,
                     int* out_rank,
                     std::string* out_err);

// ----------------------------------------------------------------
// GradCommEngine：导出成 paddle.base.core.gin_overlap.GradCommEngine
// （注册在 paddle/fluid/pybind/gin_overlap_api.cc），进程内只有一个对象。
// Python 侧按名字调用这些方法，方法名和 py::arg 名字不能改。
// 约定：
//   - 返回 int 返回码：0 成功，<10000 = cudaError_t，>=10000 = 10000 +
//     ncclResult_t。多个返回值走出参，gin_overlap_api.cc 包成 tuple。
//   - 指针按 uintptr_t 传；group = GROUP_DENSE(0) / GROUP_MOE(1)；
//     offset 是对应 window 里的字节偏移；numel / shard_numel 是元素数。
//   - send_window_tensor / copy_2d（实现在 paddle_tensor.cc）出错时抛异常，
//     不返回错误码。
class GradCommEngine {
 public:
  GradCommEngine() = default;

  static const char* error_string(int code);
  static int nccl_runtime_version();
  static int nccl_build_version();

  // 取 Paddle 已有的 comm。ring id < 0 = 不用这个组（没有专家桶）。
  // 取不到，或组大小 / 序号和传入的不一致时返回错误，err 是原因
  int set_comms(int dense_ring_id,
                int expert_ring_id,
                int device_id,
                int dense_size,
                int dense_rank,
                int expert_size,
                int expert_rank,
                std::string& err);
  int alloc_windows(int group,
                    size_t send_bytes,
                    size_t recv_bytes,
                    size_t signal_bytes);
  int create_dev_comm(int group,
                      int num_gin_contexts_hint,
                      int* num_contexts,
                      int* lsa_rank,
                      int* lsa_size);
  int set_group_topology(int group, int num_nodes, int node_index);
  // 桶数只在这里传入：建侧流、每桶 event、窗口 mark 环，并收下 host window
  // 里三块 credit 区的字节偏移（dense 组的 credit_intra / credit_inter，
  // 专家组的 credit_moe）
  int configure(int num_dense_buckets,
                int num_moe_buckets,
                int num_window_marks,
                size_t credit_intra_offset,
                size_t credit_inter_offset,
                size_t credit_moe_offset);

  int alloc_host_counters(int group,
                          size_t want_bytes,
                          uintptr_t* cpu_ptr,
                          size_t* nbytes);
};
}  // namespace gin_overlap
