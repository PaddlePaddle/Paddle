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

// gin_engine.h —— 这个扩展模块的总声明：全局状态（window / 流 / event /
// 各种偏移）、错误检查宏、
//   各阶段的 host 函数、放行层（主线程 + 尾部线程）的接口，以及导出给 Python 的
//   GradCommEngine 类。
//
// 这套代码干的事：sharding stage1 里每个梯度桶本来要做一次 NCCL
// reduce_scatter，这里改成用 NCCL 的 GIN
//   单边写（put）自己分几步做，好处是每一步都能塞进反向里 DeepEP 不占网 / 不占
//   SM 的空档里。
// 读之前需要知道的几个词（和 Python 侧同名）：
//   桶 bucket    一块连续显存，若干参数的梯度拼在里面；通信以桶为单位。
//   分片 shard   一个桶按 sharding 组切开后本卡负责的那一片。
//   window       注册给 NCCL 的对称内存，对端能单边读写；每个通信组一套 send /
//   recv / signal / host window。 mark         计算流上记的 event，侧流先等它 ⇒
//   不早于窗口开始、而且梯度已经写完。 credit       接收方腾空一个槽后给发送方
//   +1 的计数；发送方看到才敢往那个槽写。
//
// dense 桶（卡 (node_index, rank_in_node) 负责「卡号 rank_in_node 那一列」的
// num_nodes 个块
//   {node * ranks_per_node + rank_in_node}）：
//   ① CE 推列   cudaMemcpyBatchAsync 把同机 ranks_per_node-1
//   张卡各自那一列推进它们机内段里本卡那一行
//               （拷贝引擎 + NVLink，不占 SM）
//               → k_signal_intra_arrive 把本桶的 seq 写进它们 host window
//               里「本桶 × 本卡」那一格
//   ② 机内求和  k_reduce_intra_node：grad 本列块 = 本节点 ranks_per_node
//   卡之和（原地，作为 GIN put 源） ③ 机间 put  本列第 peer_node 块的部分和 →
//   节点 peer_node 同卡号 rank 的机间段第 node_index 行
//               （GIN VASignalInc 打对端 host window）
//   ④ 最终求和  本 rank shard = Σ_节点 部分和 × scale → 给 num_nodes-1
//   个发送卡回 credit
// 专家桶：同卡号 moe_sharding 组 expert_group_size 个 rank，直接 GIN put 给
// expert_group_size-1 个 peer
//   → 最终求和按 ring 序求和 → 回 credit。
// Muon 2D 桶：各卡算完自己分片后 gather 到 owner —— 跨节点 GIN
// put（k_put_to_owner），同节点 CE + release store。
//
// 缓冲：**每个通信组一套** send / recv / signal 各一块 ncclMemAlloc，注册成
// NCCL_WIN_COLL_SYMMETRIC window，组内桶间按字节偏移
//   （dense 桶在 sharding 组那套里从 0 排，专家桶在 moe_sharding 组那套里从 0
//   排）。 send window 的切片就是各桶的 grad_storage（零拷贝）：切片在 C++ 侧用
//   paddle::from_blob 包成 paddle::Tensor 还给 Python（send_window_tensor，见
//   paddle_tensor.cc）。 远端计数（credit / 机内到达 / GIN
//   signal）落在各组自己的一块 HOST_NUMA host window 上，CPU 直接读； GPU
//   上只等「本卡已交出的东西」（计算流上的 mark、本卡侧流的
//   event），不等任何远端计数。
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
// 驱动 API 出错：打印 CUresult 再返回 cudaErrorUnknown（CUresult 和 cudaError_t
// 的编号不完全一致，不直接混用）
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
// 通信组（dense / 专家各一套 comm + window） dense 桶走 sharding 组、专家桶走
// moe_sharding 组：两个 ncclComm、两个 devcomm，每组自己的 send / recv / signal
// /
//   host window。组内寻址一律用「组内序号」（专家 peer
//   就是它在专家组里的序号），不再在一个 comm 里按 base + k*stride 换算。
// 组内机器分块：rank_in_node = rank % ranks_per_node、node_index = rank /
// ranks_per_node（要求组的 rank 列表升序、
//   每节点成员数一致，Python 侧校验，并和 devcomm 的 lsaRank / lsaSize 核对）。
enum { GROUP_DENSE = 0, GROUP_MOE = 1, NUM_GROUPS = 2 };

struct CommGroup {
  ncclComm_t comm = nullptr;
  ncclDevComm_t dev_comm{};
  int num_gin_contexts =
      0;  // 实际回读的 GIN ctx 数（put kernel 按 blockIdx % 它轮转）
  int size = 0, rank = 0;  // 组大小、本卡组内序号
  int ranks_per_node = 1;  // 一个节点上的组内成员数（= dev_comm.lsaSize）
  int rank_in_node = 0;  // 本卡在节点内的序号（= dev_comm.lsaRank）
  int num_nodes = 1;     // 组内节点数
  int node_index = 0;    // 本卡所在节点的序号
  ncclWindow_t win_send{}, win_recv{}, win_signal{}, win_host{};
  char* send = nullptr;  // 本组各桶的 grad_storage（Python 用 DLPack 换过去）
  char* recv = nullptr;  // dense 组 = [机内段][机间段]；专家组 = [专家段]
  char* signal = nullptr;  // 占位（计数实际落在 host window 的同一偏移）
  char* host = nullptr;  // host window 的 CPU 地址（本进程 cuMemMap 的
                         // VA，Python 直接读）
  size_t send_bytes = 0, host_bytes = 0;
  CUmemGenericAllocationHandle host_handle = 0;
  char* lsa_recv[16] =
      {};  // 本组内同机 lsa rank l 的 recv / signal / host / send window 起点
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
extern int g_device_id;  // 本进程用的卡（set_comms 收下，之后取当前计算流 /
                         // 包窗口张量都用它）
extern cudaStream_t g_stream_intra_node_copy;    // CE 推列
extern cudaStream_t g_stream_intra_node_reduce;  // 机内求和
extern cudaStream_t g_stream_put;                // dense put + 专家 put
extern cudaStream_t g_stream_shard_reduce;       // dense + 专家最终求和
extern std::vector<cudaEvent_t> g_event_intra_node_copy,
    g_event_intra_node_reduce, g_event_dense_put, g_event_moe_put;
extern std::vector<cudaEvent_t> g_event_dense_inter_node_reduce,
    g_event_moe_reduce;
extern std::vector<cudaEvent_t> g_window_marks;  // 窗口 mark 池（循环复用）
extern int g_window_mark_index;
extern unsigned long long g_inter_node_bytes, g_intra_node_bytes;
// credit 槽的 host window 字节偏移：机内 / 机间在 dense 组的 host
// window，专家在专家组的 host window
extern size_t g_credit_intra_offset, g_credit_inter_offset, g_credit_moe_offset;
extern int g_buckets_ready;
extern cudaStream_t g_stream_gather_put;  // 2D 桶 gather：发送方 put / CE
// 2D 桶 gather 的门流（gather_setup 里建）：只放 cudaStreamWaitEvent /
// cudaEventRecord，不放 kernel。
//   原来这条流是 Python 用 paddle.device.Stream() 建的（同样是
//   cudaStreamNonBlocking），句柄靠私有属性
//   _as_parameter_ 传进来；现在引擎自己持有，Python 只说「在门流上等这个 mark /
//   记这个 event」。
extern cudaStream_t g_stream_gather_gate;
extern std::vector<cudaEvent_t>
    g_event_gather_source;  // 每 slot：本卡最终求和之后（计算流）
extern std::vector<cudaEvent_t>
    g_event_gather_put;  // 每 slot：本卡 put 或 CE 交出之后
extern int g_gather_ready;

// ---------------------------------------------------------------- Paddle 的
// 当前计算流（paddle_tensor.cc，那边才能包含 Paddle 的头文件）：
//   paddle::GetCurrentCUDAStream(GPUPlace(g_device_id))->raw_stream()，和
//   Python 的 paddle.device.current_stream()._as_parameter_.value
//   是同一条路、同一个流。 拿不到就打印原因并返回错误码（不回退到默认流）。
int paddle_compute_stream(cudaStream_t* out);
// 当前计算流，拿不到就让调用方的方法带着错误码返回
#define GIN_COMPUTE_STREAM(var)                         \
  cudaStream_t var = nullptr;                           \
  do {                                                  \
    const int _stream_rc = paddle_compute_stream(&var); \
    if (_stream_rc) return _stream_rc;                  \
  } while (0)

// ---------------------------------------------------------------- 各阶段的
// host 函数（放行层直接调这几个） 把一个桶的某个阶段交给它那条侧流（launch
// kernel / 发 CE 拷贝 / 记 event）。只有 release.cu 里的
// release_next_*（主线程的 advance 和尾部线程共用的那一份）调它们，「等哪个
// mark」是传进来的 event：主线程放行时传窗口 mark 环
// g_window_marks[mark]，尾部线程传它自己的 mark 环。
int submit_dense_intra_node_copy(int bucket,
                                 size_t recv_offset,
                                 size_t send_offset,
                                 long long shard_numel,
                                 cudaEvent_t mark_event,
                                 size_t intra_arrive_offset,
                                 unsigned long long seq);
int submit_dense_intra_node_reduce(int bucket,
                                   size_t recv_offset,
                                   size_t send_offset,
                                   long long shard_numel,
                                   int num_blocks,
                                   cudaEvent_t mark_event);
int submit_dense_inter_node_put(int bucket,
                                size_t recv_offset,
                                size_t send_offset,
                                size_t signal_offset,
                                long long shard_numel,
                                int num_blocks,
                                cudaEvent_t mark_event,
                                int* num_signals);
// 专家桶那两个阶段的组大小 / 本卡组内序号不从调用方传：直接读
// expert_group()（set_comms 时和 Python 侧核对过）
int submit_moe_put(int bucket,
                   size_t recv_offset,
                   size_t send_offset,
                   size_t signal_offset,
                   long long shard_numel,
                   int num_blocks,
                   cudaEvent_t mark_event,
                   int* num_signals);
int submit_dense_inter_node_reduce(int bucket,
                                   size_t recv_offset,
                                   size_t send_offset,
                                   long long shard_numel,
                                   float scale,
                                   int num_blocks,
                                   cudaEvent_t mark_event);
int submit_moe_reduce(int bucket,
                      size_t recv_offset,
                      size_t send_offset,
                      long long shard_numel,
                      float scale,
                      int num_blocks,
                      cudaEvent_t mark_event);
// 2D 桶 gather 的发送：跨节点 k_put_to_owner / 同节点 CE +
// k_store_release。group = GROUP_DENSE / GROUP_MOE（owner 是
//   这个组里的组内序号），wait_event = g_stream_gather_put 先等的本卡 event
int gather_put_to_owner(int group,
                        int slot,
                        size_t send_offset,
                        long long shard_numel,
                        int owner,
                        size_t arrive_offset,
                        size_t intra_arrive_offset,
                        unsigned long long seq,
                        int* num_signals,
                        cudaEvent_t wait_event);

// ---------------------------------------------------------------- 放行层
// （release.cu）：「这个桶的这个阶段现在能不能交给
// GPU」只在那里判断一次，主线程（advance）和
//   尾部线程用的是同一份判断和同一份状态。
// 阶段编号（Python 侧 RELEASE_STAGE 要解，数值不许变）
namespace release {
enum {
  STAGE_DENSE_INTRA_NODE_COPY = 1,
  STAGE_DENSE_INTRA_NODE_REDUCE = 2,
  STAGE_DENSE_INTER_NODE_PUT = 3,
  STAGE_MOE_PUT = 4,
  STAGE_DENSE_INTER_NODE_REDUCE = 5,
  STAGE_MOE_REDUCE = 6
};
enum {
  STAGE_GATHER_DENSE = 10,
  STAGE_GATHER_MOE = 11
};  // 2D 桶 gather put（Muon）
// advance 的阶段集合（Python 侧按窗口选，数值不许变）：窗口 A / B（DeepEP
// 刚结束，网卡空）交
//   机内拷贝 + put，pre 窗口（计算流在等 DeepEP，SM 空）交机内求和 + 最终求和。
enum {
  ADVANCE_INTRA_NODE_COPY = 1,
  ADVANCE_INTRA_NODE_REDUCE = 2,
  ADVANCE_PUT = 4,
  ADVANCE_SHARD_REDUCE = 8
};

int setup(const std::vector<int>& dense_order,
          const std::vector<int>& moe_order,
          const std::vector<long long>& dense_table,
          const std::vector<long long>& moe_table,
          int put_blocks,
          size_t intra_arrive_offset,
          int num_marks);
int step_reset(long long step);
int advance(cudaEvent_t mark_event,
            int stages,
            int intra_node_reduce_blocks,
            int shard_reduce_blocks,
            std::vector<int>& released_shard_reduce);
int bucket_ready(int kind, int bucket, float scale, int mark);
int record_tail_mark(cudaStream_t stream);
int tail_start(int intra_node_reduce_blocks,
               int shard_reduce_blocks,
               int mark,
               int all_posted);
int set_all_posted();
int wait_for_step_end(double timeout_s);
int abort_step(double wait_s);
int tail_finish();
int status(std::vector<int>& positions6,
           std::vector<int>& counts4,
           std::vector<int>& posted4,
           int* done);
int stuck_report(std::vector<long long>& flat);
int get_last_error(int* out3);
int get_current_activity();
// 导出给 Python 的常量表（名字 → 数值）：ADVANCE_* / STAGE_* / COUNTER_* /
// TAIL_*。数值只在 release.cu 里定义一份，Python 侧按名字取，不许自己再写数值
int constants(std::vector<std::string>& names, std::vector<int>& values);
int setup_gather_table(const std::vector<long long>& table);
int gather_put_entry(int entry_index, unsigned long long seq, int* num_signals);
int set_gather_sequences(const std::vector<unsigned long long>& seqs);
int get_gather_sent(std::vector<long long>& flat);
}  // namespace release

// ---------------------------------------------------------------- Paddle 的
// ncclComm（paddle_comm.cc） 从 Paddle 的 ProcessGroup（ring id = Python
// 侧通信组对象的 .id）取已经建好的 ncclComm_t；同时回读这个组的 大小 /
// 本卡序号。返回非 0 时 *out_err 是原因（上层报错，不回退到自己建 comm）。
int paddle_nccl_comm(int ring_id,
                     int device_id,
                     ncclComm_t* out_comm,
                     int* out_size,
                     int* out_rank,
                     std::string* out_err);

// ---------------------------------------------------------------- 导出给
// Python（paddle/fluid/pybind/gin_overlap_api.cc）
// Python 侧 paddle.distributed.communication.gin_overlap.engine
// 的 device_engine() 持有这个对象、按名字调这些方法，所以方法名和
// gin_overlap_api.cc 里的
//   py::arg 名字不能随便改（改一边要改另一边）；PaddleFleet 的放行时机层
//   （gin_overlap/scheduler.py、backend.py）也按名字调其中一部分。
// 约定：所有方法返回一个 int 返回码（0 成功，<10000 = cudaError_t，>=10000 =
// 10000 + ncclResult_t）；要多返回
//   几个值就走出参，gin_overlap_api.cc 再包成 tuple。指针按 uintptr_t 传。
//   group 参数 = GROUP_DENSE(0) / GROUP_MOE(1)；offset 一律是对应 window
//   里的字节偏移；shard_numel 是元素数。
// 流不从 Python 传：要排在 Paddle 当前计算流上的方法自己调
// paddle_compute_stream（和 Python 的
//   paddle.device.current_stream() 同一个流），要排在 2D gather 门流上的是
//   gather_gate / gather_wait_bucket / gather_mark_ready 这三个（流归引擎，见
//   g_stream_gather_gate）。
// 另有两个方法收 / 还 paddle::Tensor（实现在 paddle_tensor.cc，注册和别的
// 方法一起在 gin_overlap_api.cc）：
//   send_window_tensor(group, send_offset, numel) → 本组 send window
//   切片的 fp32 paddle::Tensor（零拷贝）； copy_2d(dst, dst_offset, dst_pitch,
//   src, src_offset, src_pitch, width, height) → 计算流上一次 pitched 拷贝。
//   这两个按 Python 的习惯直接抛异常（不是返回码）。
// 调用顺序（Python 侧 GinOverlapEngine.__init__ 里按这个顺序走一遍）：
//   set_comms → alloc_windows → alloc_host_counters → create_dev_comm →
//   set_group_topology → configure →（Muon 才有）gather_setup →
//   release_setup。
// 之后每步反向：record_window_mark 划窗口 → advance 在窗口里放行 →
// 尾部交给 tail_*（tail_mark / tail_start / bucket_ready / tail_all_posted /
//   tail_wait / tail_finish）→ join_bucket_streams + end_step →
//   release_step_reset 进下一步。 放行到哪儿了、卡在哪，用 release_status /
//   release_stuck_report 查（见 release.cu）。
// 专家桶那两个阶段（submit_moe_put / submit_moe_reduce）不带 intra_node /
// inter_node：专家桶没有机内那一步
//   （直接 put 给组内每个 peer），而且 peer 不保证跨机（专家组的两个 rank
//   可能在同一台机器上， 那笔 put 走机内），写成 inter_node 会不准。
class GradCommEngine {
 public:
  GradCommEngine() = default;

  static const char* error_string(int code);
  static int nccl_runtime_version();
  static int nccl_build_version();

  // 复用 Paddle 已有的 comm：ring id < 0 = 这个组不用（没有专家桶）。拿不到 /
  // 组大小或序号和 Python 侧不一致都报错
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
  // 桶数只在这里收一次：侧流 / 每桶 event / 窗口 mark 环 + host window 里三块
  // credit 的字节偏移（dense 组的 credit_intra / credit_inter、专家组的
  // credit_moe）。之后 release_setup 不再带桶数（它从 g_event_* 的长度读）
  int configure(int num_dense_buckets,
                int num_moe_buckets,
                int num_window_marks,
                size_t credit_intra_offset,
                size_t credit_inter_offset,
                size_t credit_moe_offset);
  int record_window_mark();

  int alloc_host_counters(int group,
                          size_t want_bytes,
                          uintptr_t* cpu_ptr,
                          size_t* nbytes);
  int join_bucket_streams(int is_dense, int bucket);
  int scale_range(int group, size_t send_offset, long long numel, float scale);
  int end_step();
  void stats(unsigned long long* out3);

  // ---- 放行层（release.cu）的 Python 入口
  int release_setup(const std::vector<int>& dense_order,
                    const std::vector<int>& moe_order,
                    const std::vector<long long>& dense_table,
                    const std::vector<long long>& moe_table,
                    int put_blocks,
                    size_t intra_arrive_offset,
                    int num_marks);
  int release_step_reset(long long step);
  int advance(int mark,
              int stages,
              int intra_node_reduce_blocks,
              int shard_reduce_blocks,
              std::vector<int>& released_shard_reduce);
  int bucket_ready(int kind, int bucket, float scale, int mark);
  int release_status(std::vector<int>& positions,
                     std::vector<int>& counts,
                     std::vector<int>& posted,
                     int* done);
  int release_stuck_report(std::vector<long long>& flat);
  int release_error(int* code, int* stage, int* bucket);
  int release_activity();
  int tail_mark();
  int tail_start(int intra_node_reduce_blocks,
                 int shard_reduce_blocks,
                 int mark,
                 int all_posted);
  int tail_all_posted();
  int tail_wait(double timeout_s);
  int tail_abort(double wait_s);
  int tail_finish();
  int tail_gather_setup(const std::vector<long long>& table);
  int tail_gather_set_sequences(const std::vector<unsigned long long>& seqs);
  int tail_gather_get_sent(std::vector<long long>& flat);

  int gather_setup(int num_slots);
  // 下面三个排在 2D gather 的门流上（g_stream_gather_gate）
  int gather_gate(int mark);
  int gather_wait_bucket(int is_dense, int bucket);
  int gather_mark_ready(int slot);
  // 主线程发一个 2D 桶的 gather put：只给表项下标和本步序号，别的参数（slot /
  // 偏移 / owner）由放行层按表查（和尾部线程用的是同一张表，见
  // release::setup_gather_table）
  int gather_put_entry(int entry_index,
                       unsigned long long seq,
                       int* num_signals);
  int gather_join(int slot);
};
}  // namespace gin_overlap
