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

// release.cu —— 放行层：判断一个桶的某个阶段现在能不能提交到 GPU，能就调
// gin_engine.cc 的 submit_* 提交。术语和约束见 gin_engine.h 文件头。
//
// 放行者有两个，共用同一份判断（release_next_*）和同一份每步状态：
//   - 主线程：每个窗口里调 advance()，按调用方给的阶段集合把能放的都放，
//     然后立即返回。
//   - 尾部线程：本步最后一个窗口之后，主线程调 tail_start() 把放行交给它；
//     它循环调 release_next_*，直到本步全部放完。
// 任何时刻只能有一个写者：尾部线程只在 RUN 状态改每步状态，advance() 只在
// IDLE 状态放行，两边都先拿 control->mutex。
//
// 交给尾部线程之后，主线程只能调：
//   - bucket_ready：某个桶的梯度凑齐了，或者换了新 mark。
//   - set_all_posted：本步所有桶都已 post。
//   - wait_for_step_end：等尾部线程离开 RUN（条件变量，不轮询）。
//   - tail_finish：尾部线程停下后收尾，回到 IDLE。
//
// mark
//   - 主线程放行用窗口 mark 环（g_window_marks，循环复用）。
//   - 尾部线程用单独的 tail_marks 环：每步从 0 号槽顺序使用，一步内一个槽只记
//     一次，用满时报错，不能绕回；下标在 step_reset 清零。
//
// 调用方（Python 侧）决定什么时候推进、每个窗口试哪些阶段、求和用多少 block；
// 这里决定每个桶能不能放并提交到 GPU。查放行进度用 status() /
// stuck_report()。
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <sys/prctl.h>

#include "paddle/fluid/distributed/collective/gin_overlap/gin_engine.h"

namespace gin_overlap {
namespace release {
namespace {

// 尾部线程的状态：
//   - IDLE：主线程放行（包括步与步之间）。
//   - RUN：尾部线程放行。
//   - DONE / ERR / ABORTED：本步放完 / 出错停下 / 被叫停。
// Python 侧按名字（TAIL_*）取这些数值。
enum { IDLE = 0, RUN = 1, DONE = 2, ERR = 3, ABORTED = 4 };
// 尾部线程既没有进展也没有新消息时，在条件变量上最多等这么久（微秒），再重新
// 读远端计数。远端计数的变化不会唤醒线程，只能轮询。
const int POLL_INTERVAL_US = 10;

// 主线程发给尾部线程的消息。kind：0 = dense 桶 bucket 的梯度凑齐了；1 = 专家
// 桶 bucket 凑齐了；-1 = 没有新桶，只换 mark。scale = 这个桶分片求和时乘的
// 系数。
struct ReadyMessage {
  int kind, bucket, mark;
  float scale;
};
// 一个 dense 桶的固定参数。字段顺序和 setup 收到的扁平表一致：
//   [send_offset, signal_offset, shard_numel, recv_offset_intra,
//    recv_offset_inter, intra_depth, inter_depth]
//   - *_offset：本组 window 里的字节偏移。
//   - shard_numel：本卡分片的元素数。
//   - *_depth：本桶和接收池这一段里上一个占用同一槽的桶之间，在放行序号上
//     相差多少（见 credit_target）。-1 = 本桶在这一段没有槽，不用等 credit。
struct DenseBucket {
  long long send_offset, signal_offset, shard_numel, recv_offset_intra,
      recv_offset_inter, intra_depth, inter_depth;
};
// 一个专家桶的固定参数：[send_offset, signal_offset, shard_numel,
// recv_offset_moe, moe_depth]，含义同 DenseBucket。
struct MoeBucket {
  long long send_offset, signal_offset, shard_numel, recv_offset_moe, moe_depth;
};

// 主线程和尾部线程共用的锁、两个条件变量和消息队列。这把锁也保护下面的每步
// 放行状态。对象 new 出来后不能释放：进程退出时尾部线程可能还在等 cv。
struct TailControl {
  std::mutex mutex;
  std::condition_variable cv_proxy,
      cv_main;  // cv_proxy 唤醒尾部线程，cv_main 唤醒等结果的主线程
  std::deque<ReadyMessage> queue;
  int state = IDLE;
  bool all_posted = false, abort = false, started = false;
  int set_device_rc = 0;  // 尾部线程里 cudaSetDevice 的返回码
};
TailControl* control = nullptr;
int device_id = 0;

// ---- setup 时设定，之后只读
//   put_blocks 由 setup 传入，没有默认值。专家组大小和本卡组内序号直接读
//   expert_group()。
int num_dense_buckets = 0, num_moe_buckets = 0, put_blocks = 0;
// host window 里各计数区的位置：intra_arrive_offset 是字节偏移，credit_*_word
// 是 u64 下标（= g_credit_*_offset / 8）。机内到达区和机内 / 机间 credit 在
// dense 组的 host window，专家 credit 在专家组的 host window。
size_t intra_arrive_offset = 0, credit_intra_word = 0, credit_inter_word = 0,
       credit_moe_word = 0;
// 固定放行顺序：*_release_order[位置] = 桶号，*_pos_in_order[桶号] = 位置
// （互为反函数）。credit 目标按位置算（见 credit_target）。
std::vector<int> dense_release_order, moe_release_order, dense_pos_in_order,
    moe_pos_in_order;
std::vector<DenseBucket> dense_buckets;
std::vector<MoeBucket> moe_buckets;
std::vector<cudaEvent_t> tail_marks;  // 尾部 mark 环（见文件头）
int tail_mark_index = 0;  // 本步已用的槽数；只有主线程改

// ---- 每步的放行状态（主线程在 IDLE 时改，尾部线程在 RUN 时改）
long long step = 0;  // 步号，credit 目标要用；由 step_reset 设定
// 各阶段已放行到固定顺序里的哪个位置：
//   - pos_dense_copy：dense 机内拷贝。
//   - stage_positions[0..2]：dense 机内求和 / put / 分片求和。
//   - pos_moe_put：专家 put。
//   - stage_positions[3]：专家分片求和。
// 一个阶段的待放行队列 = 它的位置到上一阶段位置之间的那些桶。
int pos_dense_copy = 0, stage_positions[4] = {0, 0, 0, 0}, pos_moe_put = 0;
// 本步哪些桶的梯度已经凑齐（调过 bucket_ready）。只在 step_reset 时清零。
std::vector<unsigned char> ready_dense, ready_moe;
// 下面三个跨步累加，step_reset 不清零：
//   - dense_copy_seq[桶]：本桶已提交的机内拷贝次数，写进接收卡的机内到达格。
//   - dense_targets / moe_targets[桶]：本卡为本桶累计发出的 put 笔数。各卡
//     对称，所以它也是本卡 signal 格应达到的值；分片求和等它。
std::vector<unsigned long long> dense_copy_seq, dense_targets, moe_targets;
std::vector<float> dense_scales, moe_scales;  // 各桶分片求和时乘的系数
// 机内求和 / 分片求和 kernel 的 block 数，每次 advance / tail_start 由调用方
// 传入，没有默认值
int intra_node_reduce_blocks = 0, shard_reduce_blocks = 0;
// 本次提交要等的 mark：主线程放行时是 advance 传入的窗口 mark；尾部线程放行时
// 是最近一条消息带的尾部 mark（刚交接时是 tail_start 传入的）
cudaEvent_t current_mark = nullptr;
// 尾部线程第一次看到「all_posted 且消息队列为空」时的六个位置和四个阶段的
// 待放行桶数。posted_snapshot[0] = -1 表示本步还没到这一刻。只用于日志。
int posted_snapshot[6], posted_counts[4];
// 放行出错时记下：CUDA 返回码 / 阶段编号 / 桶号
int err_rc = 0, err_stage = 0, err_bucket = -1;
// current_activity 的编码倍率：阶段编号 * ACTIVITY_STAGE_SCALE + 桶号 + 1。
// Python 侧按名字取这个数来解码
constexpr int ACTIVITY_STAGE_SCALE = 1000000;
// 正在做的提交：阶段 * ACTIVITY_STAGE_SCALE + 桶号 + 1；0 = 不在 CUDA 调用里
std::atomic<int> current_activity{0};

// 读本卡 host window 的第 word_index 个 u64（acquire load，每次都从内存读）。
// dense 桶的计数在 sharding 组的 host window，专家桶的在 moe_sharding 组的
inline unsigned long long host_counter(const CommGroup& comm_group,
                                       size_t word_index) {
  return __atomic_load_n(
      (const unsigned long long*)comm_group.host + word_index,
      __ATOMIC_ACQUIRE);
}
inline unsigned long long dense_host_counter(size_t word_index) {
  return host_counter(dense_group(), word_index);
}
inline unsigned long long moe_host_counter(size_t word_index) {
  return host_counter(expert_group(), word_index);
}

// 接收池某一段的 credit 目标值。第 step 步放行顺序第 position 个桶的序号是
// step * 桶数 + position；它要用的槽上一个占用者的序号小 depth。对端每用完一个
// 桶发一次 credit，credit 跨步累加，所以目标 = step * 桶数 + position - depth
// + 1（小于 0 时取 0）。depth < 0 = 这一段没有槽，不用等。
inline unsigned long long credit_target(long long num_buckets,
                                        long long position,
                                        long long depth) {
  if (depth < 0) return 0;
  const long long target = step * num_buckets + position - depth + 1;
  return target > 0 ? (unsigned long long)target : 0ull;
}

// 下面六个 blocked_* 是各阶段的远端放行条件，只读本卡 host window 里的计数。
// 返回 true = 还不能放。out 非空时填入等的是哪个计数、等谁、当前值和目标值
// （stuck_report 用）。
struct Blocker {
  int counter;  // COUNTER_*：等的是哪个计数（0 = 没卡住）
  int sender;   // 第几个发送方（只有机内到达格用；其余 -1）
  int peer;  // 对端的组内序号（-1 = 所有发送方共用的那一格）
  unsigned long long value, target;  // 读到的值 / 目标值
};
enum {
  COUNTER_INTRA_CREDIT = 1,  // 机内 credit
  COUNTER_INTRA_ARRIVE = 2,  // 机内到达（某个发送方那一格）
  COUNTER_INTER_CREDIT = 3,  // 机间 credit
  COUNTER_MOE_CREDIT = 4,    // 专家 credit
  COUNTER_INTER_SIGNAL = 5,  // 机间 signal（所有发送方合计）
  COUNTER_MOE_SIGNAL = 6     // 专家 signal（所有发送方合计）
};

// 机内拷贝能不能放：同节点其余各卡发来的机内 credit（[credit_intra + 8l]）都已
// 达到目标，即它们机内段里本桶要用的槽已经空出来
bool blocked_intra_node_copy(int bucket, Blocker* out) {
  const CommGroup& comm_group = dense_group();
  const unsigned long long target =
      credit_target(num_dense_buckets,
                    dense_pos_in_order[bucket],
                    dense_buckets[bucket].intra_depth);
  for (int l = 0; l < comm_group.ranks_per_node; ++l) {
    if (l == comm_group.rank_in_node) continue;
    const unsigned long long value = dense_host_counter(credit_intra_word + l);
    if (value < target) {
      if (out)
        *out = Blocker{COUNTER_INTRA_CREDIT,
                       -1,
                       comm_group.node_index * comm_group.ranks_per_node + l,
                       value,
                       target};
      return true;
    }
  }
  return false;
}
// 机内求和能不能放：本桶的机内到达格里，同节点其余每张发送卡那一格
// （[intra_arrive + (桶*16 + l)*8]）都已达到本卡本桶的机内拷贝次数
bool blocked_intra_node_reduce(int bucket, Blocker* out) {
  const CommGroup& comm_group = dense_group();
  const unsigned long long target = dense_copy_seq[bucket];
  const size_t first_word = intra_arrive_offset / 8 + (size_t)bucket * 16;
  for (int l = 0; l < comm_group.ranks_per_node; ++l) {
    if (l == comm_group.rank_in_node) continue;
    const unsigned long long value = dense_host_counter(first_word + l);
    if (value < target) {
      if (out)
        *out = Blocker{COUNTER_INTRA_ARRIVE,
                       l,
                       comm_group.node_index * comm_group.ranks_per_node + l,
                       value,
                       target};
      return true;
    }
  }
  return false;
}
// dense put 能不能放：其余各节点发来的机间 credit（[credit_inter + 8n]）都已
// 达到目标
bool blocked_dense_inter_node_put(int bucket, Blocker* out) {
  const CommGroup& comm_group = dense_group();
  const unsigned long long target =
      credit_target(num_dense_buckets,
                    dense_pos_in_order[bucket],
                    dense_buckets[bucket].inter_depth);
  for (int n = 0; n < comm_group.num_nodes; ++n) {
    if (n == comm_group.node_index) continue;
    const unsigned long long value = dense_host_counter(credit_inter_word + n);
    if (value < target) {
      if (out)
        *out = Blocker{COUNTER_INTER_CREDIT,
                       -1,
                       n * comm_group.ranks_per_node + comm_group.rank_in_node,
                       value,
                       target};
      return true;
    }
  }
  return false;
}
// 专家 put 能不能放：专家组其余各卡发来的专家 credit（[credit_moe + 8k]）都已
// 达到目标
bool blocked_moe_put(int bucket, Blocker* out) {
  const CommGroup& comm_group = expert_group();
  const unsigned long long target = credit_target(
      num_moe_buckets, moe_pos_in_order[bucket], moe_buckets[bucket].moe_depth);
  for (int k = 0; k < comm_group.size; ++k) {
    if (k == comm_group.rank) continue;
    const unsigned long long value = moe_host_counter(credit_moe_word + k);
    if (value < target) {
      if (out) *out = Blocker{COUNTER_MOE_CREDIT, -1, k, value, target};
      return true;
    }
  }
  return false;
}
// 分片求和能不能放：本桶的 signal 格已达到 dense_targets / moe_targets，即其余
// 各卡的 put 都已到达本卡
bool blocked_dense_inter_node_reduce(int bucket, Blocker* out) {
  const unsigned long long value =
      dense_host_counter((size_t)dense_buckets[bucket].signal_offset / 8);
  if (value >= dense_targets[bucket]) return false;
  if (out)
    *out = Blocker{COUNTER_INTER_SIGNAL, -1, -1, value, dense_targets[bucket]};
  return true;
}
bool blocked_moe_reduce(int bucket, Blocker* out) {
  const unsigned long long value =
      moe_host_counter((size_t)moe_buckets[bucket].signal_offset / 8);
  if (value >= moe_targets[bucket]) return false;
  if (out)
    *out = Blocker{COUNTER_MOE_SIGNAL, -1, -1, value, moe_targets[bucket]};
  return true;
}

// 记下出错位置，返回 -1，调用方据此停止本步放行
int record_failure(int rc, int stage, int bucket) {
  err_rc = rc;
  err_stage = stage;
  err_bucket = bucket;
  return -1;
}

// ---- Muon 2D 桶 gather。尾部线程放完一个 2D 桶的分片求和后，立即把本卡分片
// 发给 owner。主线程放的分片求和不在这里发 gather put，由调用方在窗口里调
// gather_put_entry 发。
//   - 表由 setup_gather_table 建；没建表时 gather_enabled = 0，不走这个分支。
//   - 每个表项对应一个 2D 桶：kind、bucket、slot（event 下标）、本卡是不是
//     owner、owner 的组内序号、本卡分片在 send window 的偏移 send_offset 和
//     元素数 shard_numel、owner host window 上的到达格（跨节点用
//     arrive_offset，同节点用 intra_arrive_offset）。
//   - gather_seq[表项]：本步这一片的 put 序号（同节点时写进到达格的值），由
//     set_gather_sequences 在交接前写入。0 = 本步不由尾部线程发。
//   - gather_sent / gather_sent_seq：本步尾部线程发了哪些表项、用了什么序号，
//     由 get_gather_sent 读出。
//   - 这些状态和其它每步状态一样，只有尾部线程不在 RUN 时主线程才能读写。
struct GatherEntry {
  int kind, bucket, slot, is_owner, owner_rank;
  long long send_offset, shard_numel;
  size_t arrive_offset, intra_arrive_offset;
};
std::vector<GatherEntry> gather_entries;
std::vector<int> gather_index_dense,
    gather_index_moe;  // 桶号 → 表项下标，-1 = 不是 2D 桶
std::vector<unsigned long long> gather_seq;
std::vector<int> gather_sent;
std::vector<unsigned long long> gather_sent_seq;
int gather_enabled = 0;

// 尾部线程在一个桶的分片求和提交到 GPU 之后调。不是 2D 桶、或本卡是 owner 时
// 什么都不做；否则让 g_stream_gather_put 等本桶分片求和的 event，再调
// gather_put_to_owner，不等远端条件。返回 0 或 CUDA 错误码。
int send_gather_put_after_shard_reduce(int kind, int bucket) {
  const int entry_index =
      (kind ? gather_index_moe : gather_index_dense)[bucket];
  if (entry_index < 0) return 0;
  const GatherEntry& entry = gather_entries[entry_index];
  if (entry.is_owner) return 0;
  const unsigned long long seq = gather_seq[entry_index];
  if (seq == 0)
    return (int)cudaErrorInvalidValue;  // 本步没给序号，或本步已经发过
  const int group = kind ? GROUP_MOE : GROUP_DENSE;
  int num_signals = 0;
  const int rc =
      gather_put_to_owner(group,
                          entry.slot,
                          (size_t)entry.send_offset,
                          entry.shard_numel,
                          entry.owner_rank,
                          entry.arrive_offset,
                          entry.intra_arrive_offset,
                          seq,
                          &num_signals,
                          kind ? g_event_moe_reduce[bucket]
                               : g_event_dense_inter_node_reduce[bucket]);
  if (rc) return rc;
  // 核对 signal 笔数：同节点走 CE 不发 signal（0），跨节点一笔 put 带一个（1）
  const CommGroup& comm_group = group_at(group);
  if (num_signals !=
      ((entry.owner_rank / comm_group.ranks_per_node == comm_group.node_index)
           ? 0
           : 1)) {
    return (int)cudaErrorInvalidValue;
  }
  gather_sent.push_back(entry_index);
  gather_sent_seq.push_back(seq);
  gather_seq[entry_index] = 0;
  return 0;
}

// 下面四个 release_next_* 各管一个阶段：从固定顺序里下一个还没放的桶开始，
// 能放的都放，遇到第一个不满足条件的桶就停（不能跳过它放后面的桶）。
// 主线程和尾部线程都调这四个函数，区别只在 current_mark 和 block 数。
// 返回 1 = 放了至少一个，0 = 一个都没放，-1 = CUDA 调用出错（已记入 err_*）。
int release_next_intra_node_copy() {
  int moved = 0;
  while (pos_dense_copy < num_dense_buckets) {
    const int bucket = dense_release_order[pos_dense_copy];
    if (!ready_dense[bucket] || blocked_intra_node_copy(bucket, nullptr))
      break;  // 梯度还没凑齐，或接收卡的槽还没空出来
    const DenseBucket& bucket_params = dense_buckets[bucket];
    // 本卡在各接收卡机内到达区里的那一格（每张接收卡上偏移相同）
    const size_t arrive_slot_offset =
        intra_arrive_offset +
        ((size_t)bucket * 16 + dense_group().rank_in_node) * 8;
    current_activity =
        STAGE_DENSE_INTRA_NODE_COPY * ACTIVITY_STAGE_SCALE + bucket + 1;
    const int rc =
        submit_dense_intra_node_copy(bucket,
                                     (size_t)bucket_params.recv_offset_intra,
                                     (size_t)bucket_params.send_offset,
                                     bucket_params.shard_numel,
                                     current_mark,
                                     arrive_slot_offset,
                                     ++dense_copy_seq[bucket]);
    current_activity = 0;
    if (rc) return record_failure(rc, STAGE_DENSE_INTRA_NODE_COPY, bucket);
    ++pos_dense_copy;
    moved = 1;
  }
  return moved;
}
int release_next_intra_node_reduce() {
  int moved = 0;
  while (stage_positions[0] < num_dense_buckets) {
    const int bucket = dense_release_order[stage_positions[0]];
    // 要求本桶的机内拷贝已放行，而且其余卡推来的行都已到达
    if (stage_positions[0] >= pos_dense_copy ||
        blocked_intra_node_reduce(bucket, nullptr))
      break;
    const DenseBucket& bucket_params = dense_buckets[bucket];
    current_activity =
        STAGE_DENSE_INTRA_NODE_REDUCE * ACTIVITY_STAGE_SCALE + bucket + 1;
    const int rc =
        submit_dense_intra_node_reduce(bucket,
                                       (size_t)bucket_params.recv_offset_intra,
                                       (size_t)bucket_params.send_offset,
                                       bucket_params.shard_numel,
                                       intra_node_reduce_blocks,
                                       current_mark);
    current_activity = 0;
    if (rc) return record_failure(rc, STAGE_DENSE_INTRA_NODE_REDUCE, bucket);
    ++stage_positions[0];
    moved = 1;
  }
  return moved;
}
int release_next_put() {
  int moved = 0, num_signals = 0;
  while (stage_positions[1] < num_dense_buckets) {
    const int bucket = dense_release_order[stage_positions[1]];
    // 要求本桶的机内求和已放行，而且对端机间段的槽已空出来
    if (stage_positions[1] >= stage_positions[0] ||
        blocked_dense_inter_node_put(bucket, nullptr))
      break;
    const DenseBucket& bucket_params = dense_buckets[bucket];
    current_activity =
        STAGE_DENSE_INTER_NODE_PUT * ACTIVITY_STAGE_SCALE + bucket + 1;
    const int rc =
        submit_dense_inter_node_put(bucket,
                                    (size_t)bucket_params.recv_offset_inter,
                                    (size_t)bucket_params.send_offset,
                                    (size_t)bucket_params.signal_offset,
                                    bucket_params.shard_numel,
                                    put_blocks,
                                    current_mark,
                                    &num_signals);
    current_activity = 0;
    if (rc) return record_failure(rc, STAGE_DENSE_INTER_NODE_PUT, bucket);
    dense_targets[bucket] += (unsigned long long)num_signals;
    ++stage_positions[1];
    moved = 1;
  }
  while (pos_moe_put < num_moe_buckets) {
    const int bucket = moe_release_order[pos_moe_put];
    if (!ready_moe[bucket] || blocked_moe_put(bucket, nullptr))
      break;  // 梯度还没凑齐，或对端的槽还没空出来
    const MoeBucket& bucket_params = moe_buckets[bucket];
    current_activity = STAGE_MOE_PUT * ACTIVITY_STAGE_SCALE + bucket + 1;
    const int rc = submit_moe_put(bucket,
                                  (size_t)bucket_params.recv_offset_moe,
                                  (size_t)bucket_params.send_offset,
                                  (size_t)bucket_params.signal_offset,
                                  bucket_params.shard_numel,
                                  put_blocks,
                                  current_mark,
                                  &num_signals);
    current_activity = 0;
    if (rc) return record_failure(rc, STAGE_MOE_PUT, bucket);
    moe_targets[bucket] += (unsigned long long)num_signals;
    ++pos_moe_put;
    moved = 1;
  }
  return moved;
}
// 分片求和比其它阶段多两个参数：
//   - in_tail = 1（尾部线程调用）：放完一个 2D 桶的分片求和后立即发它的
//     gather put（send_gather_put_after_shard_reduce）。
//   - released 非空（主线程调用）：按放行顺序追加本次放了哪些分片求和，每个
//     两个数 [kind, 桶号]。
int release_next_shard_reduce(int in_tail, std::vector<int>* released) {
  int moved = 0;
  while (stage_positions[2] < num_dense_buckets) {
    const int bucket = dense_release_order[stage_positions[2]];
    // 要求本桶的 put 已放行，而且其余卡的 put 都已到达
    if (stage_positions[2] >= stage_positions[1] ||
        blocked_dense_inter_node_reduce(bucket, nullptr))
      break;
    const DenseBucket& bucket_params = dense_buckets[bucket];
    current_activity =
        STAGE_DENSE_INTER_NODE_REDUCE * ACTIVITY_STAGE_SCALE + bucket + 1;
    const int rc =
        submit_dense_inter_node_reduce(bucket,
                                       (size_t)bucket_params.recv_offset_inter,
                                       (size_t)bucket_params.send_offset,
                                       bucket_params.shard_numel,
                                       dense_scales[bucket],
                                       shard_reduce_blocks,
                                       current_mark);
    current_activity = 0;
    if (rc) return record_failure(rc, STAGE_DENSE_INTER_NODE_REDUCE, bucket);
    ++stage_positions[2];
    if (released) {
      released->push_back(0);  // kind 0 = dense
      released->push_back(bucket);
    }
    if (in_tail && gather_enabled) {  // 只有建了 2D 桶表才为 1
      current_activity = STAGE_GATHER_DENSE * ACTIVITY_STAGE_SCALE + bucket + 1;
      const int gather_rc = send_gather_put_after_shard_reduce(0, bucket);
      current_activity = 0;
      if (gather_rc)
        return record_failure(gather_rc, STAGE_GATHER_DENSE, bucket);
    }
    moved = 1;
  }
  while (stage_positions[3] < num_moe_buckets) {
    const int bucket = moe_release_order[stage_positions[3]];
    if (stage_positions[3] >= pos_moe_put ||
        blocked_moe_reduce(bucket, nullptr))
      break;
    const MoeBucket& bucket_params = moe_buckets[bucket];
    current_activity = STAGE_MOE_REDUCE * ACTIVITY_STAGE_SCALE + bucket + 1;
    const int rc = submit_moe_reduce(bucket,
                                     (size_t)bucket_params.recv_offset_moe,
                                     (size_t)bucket_params.send_offset,
                                     bucket_params.shard_numel,
                                     moe_scales[bucket],
                                     shard_reduce_blocks,
                                     current_mark);
    current_activity = 0;
    if (rc) return record_failure(rc, STAGE_MOE_REDUCE, bucket);
    ++stage_positions[3];
    if (released) {
      released->push_back(1);  // kind 1 = 专家
      released->push_back(bucket);
    }
    if (in_tail && gather_enabled) {  // 专家 2D 桶，同上
      current_activity = STAGE_GATHER_MOE * ACTIVITY_STAGE_SCALE + bucket + 1;
      const int gather_rc = send_gather_put_after_shard_reduce(1, bucket);
      current_activity = 0;
      if (gather_rc) return record_failure(gather_rc, STAGE_GATHER_MOE, bucket);
    }
    moved = 1;
  }
  return moved;
}
// 尾部线程的一轮：分片求和 → 机内拷贝 → 机内求和 → put 各试一次。分片求和
// 排在最前，因为它会空出接收池的槽。返回 1 = 有进展，0 = 没有，-1 = 出错。
// 主线程不调这个函数。
int release_pass_once() {
  int any_progress = 0, progress;
  if ((progress = release_next_shard_reduce(1, nullptr)) < 0) return -1;
  any_progress |= progress;
  if ((progress = release_next_intra_node_copy()) < 0) return -1;
  any_progress |= progress;
  if ((progress = release_next_intra_node_reduce()) < 0) return -1;
  any_progress |= progress;
  if ((progress = release_next_put()) < 0) return -1;
  return any_progress | progress;
}
// 四个阶段各有多少桶「已到这一阶段、还没放行」，顺序 = [机内拷贝, 机内求和,
// put, 分片求和]（put 和分片求和含 dense 和专家）。机内拷贝和专家 put 只算已
// ready 的桶。只用于日志。
void pending_counts(int* out4) {
  int ready_copy = 0, ready_moe_put = 0;
  for (int k = pos_dense_copy; k < num_dense_buckets; ++k)
    if (ready_dense[dense_release_order[k]]) ++ready_copy;
  for (int k = pos_moe_put; k < num_moe_buckets; ++k)
    if (ready_moe[moe_release_order[k]]) ++ready_moe_put;
  out4[0] = ready_copy;
  out4[1] = pos_dense_copy - stage_positions[0];
  out4[2] = stage_positions[0] - stage_positions[1] + ready_moe_put;
  out4[3] = stage_positions[1] - stage_positions[2] + pos_moe_put -
            stage_positions[3];
}
// 本步是否全部放完：六个位置都到了末尾
bool is_step_done() {
  return pos_dense_copy >= num_dense_buckets &&
         stage_positions[0] >= num_dense_buckets &&
         stage_positions[1] >= num_dense_buckets &&
         stage_positions[2] >= num_dense_buckets &&
         pos_moe_put >= num_moe_buckets &&
         stage_positions[3] >= num_moe_buckets;
}

// 尾部线程一步的主循环（RUN 期间只有本线程改每步状态）：取一条消息 → 反复
// release_pass_once 直到没有进展 → 全部放完就返回 DONE。既没有消息也没有进展
// 时，在条件变量上最多等 POLL_INTERVAL_US（新消息、all_posted、abort 都会唤醒
// 它）。CUDA 调用必须在锁外做，否则会堵住主线程发消息。
int run_step() {
  for (;;) {
    ReadyMessage message{};
    bool have = false, all_posted_seen = false;
    {
      std::lock_guard<std::mutex> lk(control->mutex);
      if (control->abort) return ABORTED;
      if (!control->queue.empty()) {
        message = control->queue.front();
        control->queue.pop_front();
        have = true;
      }
      all_posted_seen = control->all_posted && control->queue.empty();
    }
    if (have) {
      if (message.kind == 0) {
        ready_dense[message.bucket] = 1;
        dense_scales[message.bucket] = message.scale;
      } else if (message.kind == 1) {
        ready_moe[message.bucket] = 1;
        moe_scales[message.bucket] = message.scale;
      }
      current_mark = tail_marks[message.mark];
    }
    if (all_posted_seen && posted_snapshot[0] < 0) {
      // 记下 all_posted 且队列为空那一刻的六个位置和待放行桶数（日志用）
      const int snapshot[6] = {pos_dense_copy,
                               stage_positions[0],
                               stage_positions[1],
                               stage_positions[2],
                               pos_moe_put,
                               stage_positions[3]};
      memcpy(posted_snapshot, snapshot, sizeof(snapshot));
      pending_counts(posted_counts);
    }
    bool moved = false;
    for (;;) {
      const int progress = release_pass_once();
      if (progress < 0) return ERR;
      if (!progress) break;
      moved = true;
    }
    if (is_step_done()) return DONE;
    if (have || moved) continue;
    std::unique_lock<std::mutex> lk(control->mutex);
    if (control->queue.empty() && !control->abort &&
        !(control->all_posted && posted_snapshot[0] < 0))
      control->cv_proxy.wait_for(lk,
                                 std::chrono::microseconds(POLL_INTERVAL_US));
  }
}
void thread_main() {
  const cudaError_t device_err =
      cudaSetDevice(device_id);  // 和主线程同一张卡（primary context）
  // 本线程的定时器余量设为 1 µs，让 POLL_INTERVAL_US 的短等待按时醒来
  prctl(PR_SET_TIMERSLACK, 1000UL, 0UL, 0UL, 0UL);
  std::unique_lock<std::mutex> lk(control->mutex);
  control->set_device_rc = (int)device_err;
  control->started = true;
  control->cv_main.notify_all();
  if (device_err != cudaSuccess) return;
  for (;;) {
    control->cv_proxy.wait(lk, [] {
      return control->state == RUN;
    });  // 步与步之间在条件变量上等，不轮询
    lk.unlock();
    const int state = run_step();
    lk.lock();
    control->state = state;
    control->cv_main.notify_all();
  }
}

}  // namespace

// 初始化，在 configure 之后调一次：收下固定放行顺序和每桶参数表，建尾部 mark
// 环（num_marks 个槽），启动尾部线程（启动后在 IDLE 等待）。
//   - 桶数取自 g_event_dense_put / g_event_moe_put 的长度；dense_order /
//     moe_order 必须是桶号的排列，参数表长度必须和桶数一致，否则报错。
//   - dense_table 每桶 7 个数，moe_table 每桶 5 个（字段见 DenseBucket /
//     MoeBucket）。
//   - intra_arrive_offset_：dense 组 host window 里机内到达区的字节偏移。
//   - credit 区的位置取自 g_credit_*_offset。
int setup(const std::vector<int>& dense_order,
          const std::vector<int>& moe_order,
          const std::vector<long long>& dense_table,
          const std::vector<long long>& moe_table,
          int put_blocks_,
          size_t intra_arrive_offset_,
          int num_marks) {
  const int num_dense = (int)g_event_dense_put.size();
  const int num_moe = (int)g_event_moe_put.size();
  if (!g_buckets_ready || !dense_group().host || control || num_marks <= 0)
    return (int)cudaErrorNotReady;
  if (num_moe > 0 && !expert_group().host) return (int)cudaErrorNotReady;
  if ((int)dense_order.size() != num_dense ||
      (int)moe_order.size() != num_moe ||
      dense_table.size() != 7u * (size_t)num_dense ||
      moe_table.size() != 5u * (size_t)num_moe)
    return (int)cudaErrorInvalidValue;
  CUDA_CHECK(cudaGetDevice(&device_id));
  num_dense_buckets = num_dense;
  num_moe_buckets = num_moe;
  put_blocks = put_blocks_;
  intra_arrive_offset = intra_arrive_offset_;
  credit_intra_word = g_credit_intra_offset / 8;  // 字节偏移 → u64 下标
  credit_inter_word = g_credit_inter_offset / 8;
  credit_moe_word = g_credit_moe_offset / 8;
  dense_release_order = dense_order;
  moe_release_order = moe_order;
  dense_pos_in_order.assign(num_dense_buckets, -1);
  moe_pos_in_order.assign(num_moe_buckets, -1);
  // 建「桶号 → 位置」，同时校验顺序是桶号的排列
  for (int k = 0; k < num_dense_buckets; ++k) {
    if (dense_release_order[k] < 0 ||
        dense_release_order[k] >= num_dense_buckets ||
        dense_pos_in_order[dense_release_order[k]] >= 0)
      return (int)cudaErrorInvalidValue;
    dense_pos_in_order[dense_release_order[k]] = k;
  }
  for (int k = 0; k < num_moe_buckets; ++k) {
    if (moe_release_order[k] < 0 || moe_release_order[k] >= num_moe_buckets ||
        moe_pos_in_order[moe_release_order[k]] >= 0)
      return (int)cudaErrorInvalidValue;
    moe_pos_in_order[moe_release_order[k]] = k;
  }
  dense_buckets.resize(num_dense_buckets);
  for (int i = 0; i < num_dense_buckets; ++i) {
    const long long* row = dense_table.data() + 7 * (size_t)i;
    dense_buckets[i] =
        DenseBucket{row[0], row[1], row[2], row[3], row[4], row[5], row[6]};
  }
  moe_buckets.resize(num_moe_buckets);
  for (int i = 0; i < num_moe_buckets; ++i) {
    const long long* row = moe_table.data() + 5 * (size_t)i;
    moe_buckets[i] = MoeBucket{row[0], row[1], row[2], row[3], row[4]};
  }
  ready_dense.assign(num_dense_buckets, 0);
  ready_moe.assign(num_moe_buckets, 0);
  dense_copy_seq.assign(num_dense_buckets, 0);
  dense_targets.assign(num_dense_buckets, 0);
  moe_targets.assign(num_moe_buckets, 0);
  dense_scales.assign(num_dense_buckets, 1.f);
  moe_scales.assign(num_moe_buckets, 1.f);
  tail_marks.assign(num_marks, nullptr);
  for (auto& ev : tail_marks)
    CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
  for (int& count : posted_counts) count = 0;
  for (int& position : posted_snapshot) position = -1;
  control = new TailControl();
  try {
    std::thread(thread_main).detach();
  } catch (...) {
    return (int)cudaErrorUnknown;
  }
  std::unique_lock<std::mutex> lk(control->mutex);
  control->cv_main.wait(lk, [] { return control->started; });
  return control->set_device_rc;
}

// 主线程在 stream 当前位置记一个尾部 mark，返回槽号；-1 = 没 setup 或
// cudaEventRecord 失败，-2 = 本步的槽已用完
int record_tail_mark(cudaStream_t stream) {
  if (!control) return -1;
  if (tail_mark_index >= (int)tail_marks.size()) return -2;
  const int i = tail_mark_index;
  if (cudaEventRecord(tail_marks[i], stream) != cudaSuccess) return -1;
  ++tail_mark_index;
  return i;
}

// 每步开始前调：清零每步状态，设定步号 step_（= 已结束的步数，绝对值）。
// step 只用于 credit 目标（见 credit_target）。dense_copy_seq / dense_targets /
// moe_targets 跨步累加，不清零。只能在 IDLE 状态调。
int step_reset(long long step_) {
  if (!control) return (int)cudaErrorNotReady;
  std::lock_guard<std::mutex> lk(control->mutex);
  if (control->state != IDLE) return (int)cudaErrorInvalidValue;
  step = step_;
  pos_dense_copy = 0;
  pos_moe_put = 0;
  for (int& position : stage_positions) position = 0;
  std::fill(ready_dense.begin(), ready_dense.end(), 0);
  std::fill(ready_moe.begin(), ready_moe.end(), 0);
  for (int& position : posted_snapshot) position = -1;
  for (int& count : posted_counts) count = 0;
  err_rc = 0;
  err_stage = 0;
  err_bucket = -1;
  current_mark = nullptr;
  tail_mark_index = 0;  // 尾部 mark 环每步从 0 号槽开始
  return 0;
}

// 主线程在一个窗口里放行。
//   - stages：要试的阶段，ADVANCE_* 按位或。
//   - mark_event：本窗口的窗口 mark。
//   - 两个 block 数：本次提交的机内求和 / 分片求和 kernel 用多少 block。
//   - 选中的阶段按 机内拷贝 → 机内求和 → put → 分片求和 的顺序各试一次，不等
//     远端，放不了的留给下一个窗口或尾部线程。
//   - released_shard_reduce：本次放了哪些分片求和，按放行顺序，每个两个数
//     [kind, 桶号]。
//   - 返回 0 或 CUDA 错误码（出错位置用 get_last_error 取）。只能在 IDLE 状态
//     调，否则返回 cudaErrorInvalidValue。
int advance(cudaEvent_t mark_event,
            int stages,
            int intra_node_reduce_blocks_,
            int shard_reduce_blocks_,
            std::vector<int>& released_shard_reduce) {
  released_shard_reduce.clear();
  if (!control || control->set_device_rc) return (int)cudaErrorNotReady;
  std::lock_guard<std::mutex> lk(control->mutex);
  if (control->state != IDLE || !mark_event) return (int)cudaErrorInvalidValue;
  intra_node_reduce_blocks = intra_node_reduce_blocks_;
  shard_reduce_blocks = shard_reduce_blocks_;
  current_mark = mark_event;
  if (stages & ADVANCE_INTRA_NODE_COPY)
    if (release_next_intra_node_copy() < 0) return err_rc;
  if (stages & ADVANCE_INTRA_NODE_REDUCE)
    if (release_next_intra_node_reduce() < 0) return err_rc;
  if (stages & ADVANCE_PUT)
    if (release_next_put() < 0) return err_rc;
  if (stages & ADVANCE_SHARD_REDUCE)
    if (release_next_shard_reduce(0, &released_shard_reduce) < 0) return err_rc;
  return 0;
}

// 把放行交给尾部线程，状态从 IDLE 变成 RUN。只能在 IDLE 状态调。
//   - 两个 block 数：尾部线程提交的机内求和 / 分片求和 kernel 用多少 block。
//   - mark_：交接时记的尾部 mark 槽号。
//   - all_posted = 1：本步所有桶在交接前都已 post（本步没有进入尾部，在
//     reduce_gradients 里才交接）。
int tail_start(int intra_node_reduce_blocks_,
               int shard_reduce_blocks_,
               int mark_,
               int all_posted) {
  if (!control || control->set_device_rc) return (int)cudaErrorNotReady;
  std::lock_guard<std::mutex> lk(control->mutex);
  if (control->state != IDLE || mark_ < 0 || mark_ >= tail_mark_index)
    return (int)cudaErrorInvalidValue;
  intra_node_reduce_blocks = intra_node_reduce_blocks_;
  shard_reduce_blocks = shard_reduce_blocks_;
  current_mark = tail_marks[mark_];
  control->queue.clear();
  control->all_posted = all_posted != 0;
  control->abort = false;
  control->state = RUN;
  control->cv_proxy.notify_all();
  return 0;
}

// 某个桶的梯度凑齐了（kind 0 = dense 桶，1 = 专家桶），或者只换 mark（kind
// -1）。scale = 这个桶分片求和时乘的系数。
//   - IDLE 状态：mark_ 必须是 -1，直接记下 ready 和 scale，由之后的 advance
//     放行。
//   - RUN 状态：mark_ 是尾部 mark 槽号，作为一条消息发给尾部线程。
//   - 返回 0 = 收下；1 = 尾部线程已出错停下（原因用 get_last_error 取）；2 =
//     用法错误（mark 和状态不匹配、桶号无效、这个桶本步已经 ready 过、全部放完
//     后又有桶 ready）。
int bucket_ready(int kind, int bucket, float scale, int mark_) {
  if (!control) return 2;
  std::lock_guard<std::mutex> lk(control->mutex);
  if (control->state == ERR) return 1;
  if (kind > 1 ||
      (kind >= 0 &&
       (bucket < 0 || bucket >= (kind ? num_moe_buckets : num_dense_buckets))))
    return 2;
  if (control->state == IDLE) {  // 主线程还在放行：直接记下
    if (mark_ >= 0 || kind < 0 || is_step_done()) return 2;
    unsigned char& ready =
        (kind ? ready_moe : ready_dense)[bucket];  // 每步只能 ready 一次
    if (ready) return 2;
    ready = 1;
    (kind ? moe_scales : dense_scales)[bucket] = scale;
    return 0;
  }
  if (mark_ < 0 || mark_ >= tail_mark_index) return 2;
  if (control->state == DONE)
    return kind < 0 ? 0 : 2;  // 已全部放完：只换 mark 的消息直接丢弃
  if (control->state != RUN) return 2;
  control->queue.push_back(ReadyMessage{kind, bucket, mark_, scale});
  control->cv_proxy.notify_all();
  return 0;
}

int set_all_posted() {
  if (!control) return 2;
  std::lock_guard<std::mutex> lk(control->mutex);
  if (control->state == ERR) return 1;
  if (control->state != RUN && control->state != DONE) return 2;
  control->all_posted = true;
  control->cv_proxy.notify_all();
  return 0;
}

// 等尾部线程离开 RUN（放完 / 出错 / 被叫停），最多等 timeout_s 秒，返回当时
// 的状态；超时时返回 RUN。在条件变量上等，不轮询。
int wait_for_step_end(double timeout_s) {
  if (!control) return -1;
  std::unique_lock<std::mutex> lk(control->mutex);
  const auto deadline =
      std::chrono::steady_clock::now() +
      std::chrono::duration_cast<std::chrono::steady_clock::duration>(
          std::chrono::duration<double>(timeout_s));
  control->cv_main.wait_until(
      lk, deadline, [] { return control->state != RUN; });
  return control->state;
}

// 叫停尾部线程，最多等 wait_s 秒，返回之后的状态。尾部线程只在每轮开头检查
// abort；它卡在 CUDA 调用里时，这里等到超时并返回 RUN
int abort_step(double wait_s) {
  if (!control) return -1;
  {
    std::lock_guard<std::mutex> lk(control->mutex);
    if (control->state != RUN) return control->state;
    control->abort = true;
    control->cv_proxy.notify_all();
  }
  return wait_for_step_end(wait_s);
}

// 尾部线程停下之后收尾：清空消息队列和 all_posted / abort 标志，回到 IDLE。
// 只能在 DONE / ERR / ABORTED 状态调。
int tail_finish() {
  if (!control) return (int)cudaErrorNotReady;
  std::lock_guard<std::mutex> lk(control->mutex);
  if (control->state == IDLE || control->state == RUN)
    return (int)cudaErrorNotReady;
  control->state = IDLE;
  control->queue.clear();
  control->all_posted = control->abort = false;
  return 0;
}

// 本步的放行进度。不能在 RUN 状态调。
//   - positions6 = [dense 机内拷贝, dense 机内求和, dense put, dense 分片求和,
//     专家 put, 专家分片求和] 已放行到的位置。
//   - counts4 = 四个阶段的待放行桶数（顺序同 pending_counts）。
//   - posted4 = posted_snapshot 时刻的同一组数；没有这一刻时是 [-1, 0, 0, 0]。
//   - *done = 本步是否全部放完。
int status(std::vector<int>& positions6,
           std::vector<int>& counts4,
           std::vector<int>& posted4,
           int* done) {
  if (!control) return (int)cudaErrorNotReady;
  std::lock_guard<std::mutex> lk(control->mutex);
  if (control->state == RUN) return (int)cudaErrorNotReady;
  positions6 = {pos_dense_copy,
                stage_positions[0],
                stage_positions[1],
                stage_positions[2],
                pos_moe_put,
                stage_positions[3]};
  int counts[4] = {0, 0, 0, 0};
  pending_counts(counts);
  counts4.assign(counts, counts + 4);
  posted4.assign(4, 0);
  posted4[0] = -1;
  if (posted_snapshot[0] >= 0) posted4.assign(posted_counts, posted_counts + 4);
  *done = is_step_done() ? 1 : 0;
  return 0;
}

// 卡住时的现场（只读 host window，不改状态）。不能在 RUN 状态调。
// flat 里依次是六个阶段的记录（dense 四个阶段，再专家两个），每条记录：
//   - 6 个数的表头：[阶段编号 STAGE_*, 位置, 桶数, 下一个桶是否已到这一阶段
//     （1 / 0）, 待放行桶数, 后面列出的桶数 n]。
//   - n 组、每组 6 个数（n ≤ STUCK_MAX_ITEMS）：[桶号, 等的计数 COUNTER_*（0 =
//     条件已满足，只是排在前一个桶后面）, 第几个发送方（-1 = 不适用）, 对端组内
//     序号（-1 = 所有发送方共用的那一格）, 当前值, 目标值]。
const int STUCK_MAX_ITEMS = 8;

static void stuck_stage(int stage, std::vector<long long>& flat) {
  const bool is_moe = stage == STAGE_MOE_PUT || stage == STAGE_MOE_REDUCE;
  const std::vector<int>& order =
      is_moe ? moe_release_order : dense_release_order;
  const int num_buckets = is_moe ? num_moe_buckets : num_dense_buckets;
  int position = 0, previous = 0;  // previous = 上一阶段的位置（-1 = 看 ready）
  switch (stage) {
    case STAGE_DENSE_INTRA_NODE_COPY:
      position = pos_dense_copy;
      previous = -1;
      break;
    case STAGE_DENSE_INTRA_NODE_REDUCE:
      position = stage_positions[0];
      previous = pos_dense_copy;
      break;
    case STAGE_DENSE_INTER_NODE_PUT:
      position = stage_positions[1];
      previous = stage_positions[0];
      break;
    case STAGE_DENSE_INTER_NODE_REDUCE:
      position = stage_positions[2];
      previous = stage_positions[1];
      break;
    case STAGE_MOE_PUT:
      position = pos_moe_put;
      previous = -1;
      break;
    default:  // STAGE_MOE_REDUCE
      position = stage_positions[3];
      previous = pos_moe_put;
      break;
  }
  // 这个阶段的待放行队列：位置之后、已经到了这一阶段的桶（已 ready，或者上一
  // 阶段的位置在它后面）
  std::vector<int> items;
  for (int k = position; k < num_buckets; ++k) {
    const bool arrived = previous < 0
                             ? (is_moe ? ready_moe : ready_dense)[order[k]] != 0
                             : k < previous;
    if (arrived) items.push_back(order[k]);
  }
  const int listed =
      (int)items.size() < STUCK_MAX_ITEMS ? (int)items.size() : STUCK_MAX_ITEMS;
  flat.push_back(stage);
  flat.push_back(position);
  flat.push_back(num_buckets);
  flat.push_back((long long)(position < num_buckets && !items.empty() &&
                             items[0] == order[position]));
  flat.push_back((long long)items.size());
  flat.push_back(listed);
  for (int i = 0; i < listed; ++i) {
    const int bucket = items[i];
    Blocker blocker{0, -1, -1, 0, 0};
    switch (stage) {
      case STAGE_DENSE_INTRA_NODE_COPY:
        blocked_intra_node_copy(bucket, &blocker);
        break;
      case STAGE_DENSE_INTRA_NODE_REDUCE:
        blocked_intra_node_reduce(bucket, &blocker);
        break;
      case STAGE_DENSE_INTER_NODE_PUT:
        blocked_dense_inter_node_put(bucket, &blocker);
        break;
      case STAGE_DENSE_INTER_NODE_REDUCE:
        blocked_dense_inter_node_reduce(bucket, &blocker);
        break;
      case STAGE_MOE_PUT:
        blocked_moe_put(bucket, &blocker);
        break;
      default:
        blocked_moe_reduce(bucket, &blocker);
        break;
    }
    flat.push_back(bucket);
    flat.push_back(blocker.counter);
    flat.push_back(blocker.sender);
    flat.push_back(blocker.peer);
    flat.push_back((long long)blocker.value);
    flat.push_back((long long)blocker.target);
  }
}

int stuck_report(std::vector<long long>& flat) {
  flat.clear();
  if (!control) return (int)cudaErrorNotReady;
  std::lock_guard<std::mutex> lk(control->mutex);
  if (control->state == RUN) return (int)cudaErrorNotReady;
  for (const int stage : {STAGE_DENSE_INTRA_NODE_COPY,
                          STAGE_DENSE_INTRA_NODE_REDUCE,
                          STAGE_DENSE_INTER_NODE_PUT,
                          STAGE_DENSE_INTER_NODE_REDUCE,
                          STAGE_MOE_PUT,
                          STAGE_MOE_REDUCE})
    stuck_stage(stage, flat);
  return 0;
}

// out3 = [CUDA 返回码, 阶段编号（STAGE_*）, 桶号]；没出错时是 [0, 0, -1]
int get_last_error(int* out3) {
  if (!control) return (int)cudaErrorNotReady;
  std::lock_guard<std::mutex> lk(control->mutex);
  out3[0] = err_rc;
  out3[1] = err_stage;
  out3[2] = err_bucket;
  return 0;
}

// 当前正在做的提交，编码同 current_activity；0 = 不在 CUDA 调用里
int get_current_activity() { return current_activity.load(); }

// 导出给 Python 的常量表（名字 → 数值）：ADVANCE_*、STAGE_*、COUNTER_*、
// TAIL_* 和 ACTIVITY_STAGE_SCALE。gin_overlap_api.cc 把它们导出成
// paddle.base.core.gin_overlap 的模块属性。加、改、删常量时必须同时改这张表。
int constants(std::vector<std::string>& names, std::vector<int>& values) {
  names.clear();
  values.clear();
  const struct {
    const char* name;
    int value;
  } table[] = {
      {"ADVANCE_INTRA_NODE_COPY", ADVANCE_INTRA_NODE_COPY},
      {"ADVANCE_INTRA_NODE_REDUCE", ADVANCE_INTRA_NODE_REDUCE},
      {"ADVANCE_PUT", ADVANCE_PUT},
      {"ADVANCE_SHARD_REDUCE", ADVANCE_SHARD_REDUCE},
      {"STAGE_DENSE_INTRA_NODE_COPY", STAGE_DENSE_INTRA_NODE_COPY},
      {"STAGE_DENSE_INTRA_NODE_REDUCE", STAGE_DENSE_INTRA_NODE_REDUCE},
      {"STAGE_DENSE_INTER_NODE_PUT", STAGE_DENSE_INTER_NODE_PUT},
      {"STAGE_MOE_PUT", STAGE_MOE_PUT},
      {"STAGE_DENSE_INTER_NODE_REDUCE", STAGE_DENSE_INTER_NODE_REDUCE},
      {"STAGE_MOE_REDUCE", STAGE_MOE_REDUCE},
      {"STAGE_GATHER_DENSE", STAGE_GATHER_DENSE},
      {"STAGE_GATHER_MOE", STAGE_GATHER_MOE},
      {"COUNTER_INTRA_CREDIT", COUNTER_INTRA_CREDIT},
      {"COUNTER_INTRA_ARRIVE", COUNTER_INTRA_ARRIVE},
      {"COUNTER_INTER_CREDIT", COUNTER_INTER_CREDIT},
      {"COUNTER_MOE_CREDIT", COUNTER_MOE_CREDIT},
      {"COUNTER_INTER_SIGNAL", COUNTER_INTER_SIGNAL},
      {"COUNTER_MOE_SIGNAL", COUNTER_MOE_SIGNAL},
      {"TAIL_IDLE", IDLE},
      {"TAIL_RUN", RUN},
      {"TAIL_DONE", DONE},
      {"TAIL_ERR", ERR},
      {"TAIL_ABORTED", ABORTED},
      {"ACTIVITY_STAGE_SCALE", ACTIVITY_STAGE_SCALE},
  };
  for (const auto& item : table) {
    names.push_back(item.name);
    values.push_back(item.value);
  }
  return 0;
}

// 建 2D 桶表。必须在 setup 和 gather_setup 之后、IDLE 状态下调，只能调一次。
// table 每项 10 个数：
//   [kind（0 dense / 1 专家）, 桶号, slot, 本卡是不是 owner, owner 的组内序号,
//    send_offset, shard_numel, 保留位（必须 ≥ 1）, arrive_offset,
//    intra_arrive_offset]
// 表为空时不建，gather_enabled 保持 0。
int setup_gather_table(const std::vector<long long>& table) {
  if (!control || !g_gather_ready || gather_enabled)
    return (int)cudaErrorNotReady;
  if (table.size() % 10u) return (int)cudaErrorInvalidValue;
  const int num_entries = (int)(table.size() / 10u);
  std::lock_guard<std::mutex> lk(control->mutex);
  if (control->state != IDLE) return (int)cudaErrorInvalidValue;
  std::vector<GatherEntry> entries;
  std::vector<int> index_dense(num_dense_buckets, -1),
      index_moe(num_moe_buckets, -1);
  for (int i = 0; i < num_entries; ++i) {
    const long long* row = table.data() + 10 * (size_t)i;
    GatherEntry entry{(int)row[0],
                      (int)row[1],
                      (int)row[2],
                      (int)row[3],
                      (int)row[4],
                      row[5],
                      row[6],
                      (size_t)row[8],
                      (size_t)row[9]};
    if (entry.kind < 0 || entry.kind > 1 || entry.bucket < 0 ||
        entry.bucket >= (entry.kind ? num_moe_buckets : num_dense_buckets) ||
        entry.slot < 0 || entry.slot >= (int)g_event_gather_put.size() ||
        entry.shard_numel <= 0 || entry.send_offset < 0 || row[7] < 1)
      return (int)cudaErrorInvalidValue;
    int& entry_index = (entry.kind ? index_moe : index_dense)[entry.bucket];
    if (entry_index >= 0)
      return (int)cudaErrorInvalidValue;  // 同一个桶出现了两项
    entry_index = i;
    entries.push_back(entry);
  }
  gather_entries.swap(entries);
  gather_index_dense.swap(index_dense);
  gather_index_moe.swap(index_moe);
  gather_seq.assign(num_entries, 0);
  gather_sent.clear();
  gather_sent_seq.clear();
  gather_sent.reserve(num_entries);
  gather_sent_seq.reserve(num_entries);
  gather_enabled = num_entries > 0 ? 1 : 0;
  return 0;
}

// 主线程发一个 2D 桶的 gather put：按表项下标查 setup_gather_table 建的表，
// 等 g_event_gather_source[slot]（由 gather_mark_ready 记）。只能在 IDLE 状态
// 调。
//   - seq：本步这一片的 put 序号（同节点时写进 owner 的到达格），必须 > 0。
//   - *num_signals：发出的 signal 笔数（同节点 0，跨节点 1）。
int gather_put_entry(int entry_index,
                     unsigned long long seq,
                     int* num_signals) {
  *num_signals = 0;
  if (!control || !gather_enabled) return (int)cudaErrorNotReady;
  std::lock_guard<std::mutex> lk(control->mutex);
  if (control->state != IDLE || entry_index < 0 ||
      entry_index >= (int)gather_entries.size() || seq == 0)
    return (int)cudaErrorInvalidValue;
  const GatherEntry& entry = gather_entries[entry_index];
  if (entry.is_owner || entry.slot < 0 ||
      entry.slot >= (int)g_event_gather_source.size())
    return (int)cudaErrorInvalidValue;
  return gather_put_to_owner(entry.kind ? GROUP_MOE : GROUP_DENSE,
                             entry.slot,
                             (size_t)entry.send_offset,
                             entry.shard_numel,
                             entry.owner_rank,
                             entry.arrive_offset,
                             entry.intra_arrive_offset,
                             seq,
                             num_signals,
                             g_event_gather_source[entry.slot]);
}

// 设定本步每个表项的 put 序号（0 = 本步不由尾部线程发），并清空上一步的发送
// 记录。必须在 tail_start 之前、IDLE 状态下调
int set_gather_sequences(const std::vector<unsigned long long>& seqs) {
  if (!control || !gather_enabled) return (int)cudaErrorNotReady;
  std::lock_guard<std::mutex> lk(control->mutex);
  if (control->state != IDLE || seqs.size() != gather_entries.size())
    return (int)cudaErrorInvalidValue;
  gather_seq = seqs;
  gather_sent.clear();
  gather_sent_seq.clear();
  return 0;
}

// 本步尾部线程发出的 gather put，按发出顺序，每个三个数 [kind, bucket, seq]。
// 不能在 RUN 状态调
int get_gather_sent(std::vector<long long>& flat) {
  flat.clear();
  if (!control || !gather_enabled) return (int)cudaErrorNotReady;
  std::lock_guard<std::mutex> lk(control->mutex);
  if (control->state == RUN) return (int)cudaErrorInvalidValue;
  flat.reserve(gather_sent.size() * 3);
  for (size_t i = 0; i < gather_sent.size(); ++i) {
    const GatherEntry& entry = gather_entries[gather_sent[i]];
    flat.push_back(entry.kind);
    flat.push_back(entry.bucket);
    flat.push_back((long long)gather_sent_seq[i]);
  }
  return 0;
}

}  // namespace release
}  // namespace gin_overlap
