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

// release.cu —— 放行层：一个桶的某个阶段「现在能不能交给
// GPU」只在这里判断一次。
//
// 一步反向里有两个放行者，用的是同一份判断、同一份状态（下面那些每步状态）：
//   主线程     每过一个 DeepEP 不占网 / 不占 SM 的窗口，调 advance()：按 Python
//   指定的阶段集合
//              同步跑一遍下面的
//              release_next_*，能放的都放，然后立刻返回去干别的。
//   尾部线程   本步最后一个 combine
//   之后再没有窗口了，但还有桶没放完。主线程不留在那儿等，而是调
//              tail_start() 把放行交给这个后台线程（下称
//              proxy），它自己循环跑同一套 release_next_*。
// 两者不会同时跑：尾部线程只在 RUN 状态里动这些状态，advance() 只在 IDLE
// 状态下才放行（都要先拿
//   control->mutex），所以任何时刻只有一个写者，不需要在两边各存一份状态、也不需要交接时搬状态。
//
// 交接之后主线程只做四件事：
//   bucket_ready      某个桶的梯度刚凑齐（或它的 mark 换了），告诉尾部线程
//   set_all_posted    本步的桶都交出去了，不会再有新的
//   wait_for_step_end 等它放完（条件变量，不轮询）
//   tail_finish       它停下之后收尾，回到 IDLE（放到哪儿了由 status /
//   stuck_report 查）
//
// mark（GPU 上的时间点）：主线程放行用窗口 mark 环（gin_engine.cc 的
// g_window_marks，循环复用）；
//   尾部线程要等的 event 记在单独的环 tail_marks 里 ——
//   每步从 0 号槽开始顺序用、一步内一个槽只记一次、
//   用满就报错而不是绕回覆盖，下标在 step_reset 清零。这样尾部线程正在等的
//   event 不会被后来记的 mark 覆盖掉。
//
// 和 Python 侧的分工（放行时机在 PaddleFleet 的
// gin_overlap/scheduler.py）：Python
// 决定「什么时候推进、哪个窗口试哪些阶段、求和用多少
//   block」，这里决定「某个桶现在能不能放」并真的交给 GPU（gin_engine.cc 里的
//   submit_*）。 Python 侧不再有第二份放行判断，也不再持有桶队列 /
//   影子计数；要打日志或超时报告时来查 status() / stuck_report()。
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

// 尾部线程的状态机：IDLE 没在接管（主线程自己
// advance，步与步之间也是这个状态）→ RUN 正在接管放行 → DONE 本步放完 / ERR
// 出错停下 / ABORTED 被叫停。 数值和 Python 侧对应（RUN=1 / DONE=2 /
// ERR=3），不许变。
enum { IDLE = 0, RUN = 1, DONE = 2, ERR = 3, ABORTED = 4 };
// 既没有进展也没有新消息时，在条件变量上最多睡这么久再回头看一眼远端计数（计数是对端网卡
// / 对端 GPU 直接写 本卡 host window，没有谁来唤醒我们，所以这一项只能轮询；10
// µs 是「别空转烧 CPU」和「别晚放行」的折中）。
const int POLL_INTERVAL_US = 10;

// 主线程投给尾部线程的一条消息。kind：0 = dense 桶 bucket 的梯度凑齐了，1 =
// 专家桶 bucket 凑齐了，
//   -1 = 没有新桶，只是换 mark（又过了一个 DeepEP 窗口）。scale =
//   这个桶求和后要乘的系数。
struct ReadyMessage {
  int kind, bucket, mark;
  float scale;
};
// 一个 dense 桶的固定参数，字段顺序和 Python 侧交过来的扁平表一致：
//   [send_offset, signal_offset, shard_numel, recv_offset_intra,
//   recv_offset_inter, intra_depth, inter_depth]。
//   *_offset 都是本组 window 里的字节偏移，shard_numel 是元素数（float
//   个数，不是字节）。
//   *_depth = 接收池这一段能同时存几个桶（槽数）；-1 表示这一段大小为
//   0，也就不用等 credit。
struct DenseBucket {
  long long send_offset, signal_offset, shard_numel, recv_offset_intra,
      recv_offset_inter, intra_depth, inter_depth;
};
// 一个专家桶的固定参数：[send_offset, signal_offset, shard_numel,
// recv_offset_moe, moe_depth]，含义同上。
struct MoeBucket {
  long long send_offset, signal_offset, shard_numel, recv_offset_moe, moe_depth;
};

// 主线程和尾部线程之间的那把锁、两个条件变量、消息队列。锁同时保护下面的每步放行状态：
//   尾部线程在 RUN 里改它们，主线程只在 IDLE 里改（advance / bucket_ready /
//   step_reset 都拿这把锁）。 new 出来不释放：进程退出时尾部线程可能还睡在 cv
//   上，不能让静态析构先把 mutex / cv 拆掉。
struct TailControl {
  std::mutex mutex;
  std::condition_variable cv_proxy,
      cv_main;  // cv_proxy 叫醒尾部线程，cv_main 叫醒等结果的主线程
  std::deque<ReadyMessage> queue;
  int state = IDLE;
  bool all_posted = false, abort = false, started = false;
  int set_device_rc =
      0;  // 尾部线程里 cudaSetDevice 的返回码（起线程时回给主线程）
};
TailControl* control = nullptr;
int device_id = 0;

// ---- setup 时定下、之后只读
//   put_blocks 没有默认值：setup 收下 Python 侧的 engine.PUT_BLOCKS（这个数只在
//   那边写一份，这里再写一个 64 只会让人以为改它有用）。
//   专家组大小 / 本卡在专家组里的序号不在这里存第三份：要用的地方直接读
//   expert_group()（set_comms 已经把它和 Python 侧核对过、存进 CommGroup）。
int num_dense_buckets = 0, num_moe_buckets = 0, put_blocks = 0;
// host window 里几块计数区的位置。intra_arrive_offset 是字节偏移，credit_*_word
// 是 u64 下标（= 字节偏移 / 8）。
//   机内到达区和机内 / 机间 credit 在 dense 组的 host window 里，专家 credit
//   在专家组的 host window 里。三块 credit 的字节偏移不从 Python 过来第二遍：
//   setup 里由 g_credit_*_offset（上层初始化时只交一次的那一份）除 8 得到。
size_t intra_arrive_offset = 0, credit_intra_word = 0, credit_inter_word = 0,
       credit_moe_word = 0;
// 放行顺序：*_release_order[位置] = 桶号，*_pos_in_order[桶号] =
// 它在这个顺序里的位置（互为反函数）。
//   顺序是固定的，接收池按这个顺序腾空，所以能用「位置」算出该等到多少
//   credit（见 credit_target）。
std::vector<int> dense_release_order, moe_release_order, dense_pos_in_order,
    moe_pos_in_order;
std::vector<DenseBucket> dense_buckets;
std::vector<MoeBucket> moe_buckets;
std::vector<cudaEvent_t> tail_marks;  // 尾部 mark 环（见文件头）
int tail_mark_index = 0;  // 本步已用掉几个槽；只有主线程改（record_tail_mark /
                          // step_reset）

// ---- 每步的放行状态（主线程在 IDLE 里改，尾部线程在 RUN 里改，见
// TailControl）
long long step = 0;  // 第几步（credit 目标要用，step_reset 时由 Python 给）
// 每个阶段放到哪儿了（固定顺序里的下标）：pos_dense_copy =
// CE，stage_positions[0..2] = dense 的机内求和 / put /
//   最终求和，pos_moe_put = 专家 put，stage_positions[3] = 专家最终求和。
//   「某个桶在某个阶段的队列里」不单独记，就是用这些位置表示的：一个阶段的队列
//   = 它的位置和上一阶段位置之间那一段（见 pending_counts）。
int pos_dense_copy = 0, stage_positions[4] = {0, 0, 0, 0}, pos_moe_put = 0;
// 本步哪些桶的梯度已经凑齐（Python 调过 bucket_ready）。
//   和 Python 侧 engine.moe_posted / dense_posted 是同一个 bit
//   的两份，**故意**不合并： 这一份整步有效（step_reset
//   才清零），尾部线程靠它判断「还有哪些桶能放」； Python 那一份在
//   finish_bucket（reduce_gradients 里逐桶 join 之后）就清零，管的是 「本步
//   post 过没有」（post 两次直接报错）和 backend 问「哪些桶还没来」。
//   合成一份就得在两个清零时机里挑一个，另一头必然拿到过期答案。
std::vector<unsigned char> ready_dense, ready_moe;
// dense_copy_seq[桶] = 本桶已经交出过几次 CE（从 1
// 开始，写进接收卡的到达格）；跨步累加。
// dense_targets / moe_targets[桶] = 本桶已经发出多少笔 put（= 对端 signal
// 该涨到多少，最终求和等它）；也跨步累加，所以 step_reset 不清它们。
std::vector<unsigned long long> dense_copy_seq, dense_targets, moe_targets;
std::vector<float> dense_scales, moe_scales;  // 各桶求和后乘的系数
// 这两个阶段的 kernel 用多少 block：每次 advance / tail_start 都由 Python 侧给
//   （scheduler.INTRA_NODE_REDUCE_BLOCKS / SHARD_REDUCE_BLOCKS / TAIL_BLOCKS，
//   SM 预算归策略），所以这里不设默认值 —— 放行之前一定被赋过值。
int intra_node_reduce_blocks = 0, shard_reduce_blocks = 0;
cudaEvent_t current_mark =
    nullptr;  // 当前等的 mark：主线程放行时是 advance 传的窗口 mark，尾部线程
              // 放行时是最近处理的那条消息带的（交接时是交接那一刻记的）
// 尾部线程看到「桶都交出去了、队列也空了」那一刻的六个位置和四个阶段的待放行桶数；
//   posted_snapshot[0] = -1 表示本步没看到过这一刻。Python
//   用它报告「兜底时还剩多少桶」。
int posted_snapshot[6], posted_counts[4];
int err_rc = 0, err_stage = 0,
    err_bucket = -1;  // 放行出错时：CUDA 返回码 / 阶段编号 / 桶号
// 「此刻正在发哪个 CUDA 调用」的编码倍率（get_current_activity）：阶段编号 *
//   这个数 + 桶号 + 1。Python 侧用它反解（scheduler 从 release::constants
//   按名字 取 ACTIVITY_STAGE_SCALE），所以这个数也只在这里定义一份。
constexpr int ACTIVITY_STAGE_SCALE = 1000000;
std::atomic<int> current_activity{
    0};  // 正在做的 CUDA 调用：阶段 * ACTIVITY_STAGE_SCALE + 桶号 + 1；0 =
         // 不在调用里

// 读本卡 host window 的第 word_index 个
// u64。每次都重新从内存读（这些格子是对端网卡或对端 GPU 写的，
//   本卡 CPU 只读，所以不能让编译器把它缓存在寄存器里）。dense 桶的计数在
//   sharding 组那块 host window、 专家桶的在 moe_sharding 组那块。
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

// 接收池某一段该等到多少 credit：所有 rank 按同一个固定顺序腾空，所以第 step
// 步放行顺序里第 position 个桶
//   要用的那个槽，是被「depth 个桶之前的那个桶」占着的 —— 对端做完它就会 +1。
//   credit 是跨步累加的绝对值（谁也不清零），所以目标值 = step * 桶数 +
//   position - depth + 1。 depth < 0 = 这一段没有槽（不走接收池），不用等。
inline unsigned long long credit_target(long long num_buckets,
                                        long long position,
                                        long long depth) {
  if (depth < 0) return 0;
  const long long target = step * num_buckets + position - depth + 1;
  return target > 0 ? (unsigned long long)target : 0ull;
}

// 下面六个 blocked_* 是各阶段的远端放行条件，整个库里只有这一份（都只读本卡
// host window 里的计数；
//   对端网卡 / 对端 GPU 直接写这些格子，GPU 上不等任何远端计数）。返回 true =
//   还不能放； out
//   非空时顺带把现场填进去（哪个计数、等谁、读到多少、要等到多少），超时报告用（见
//   stuck_report）。
struct Blocker {
  int counter;  // COUNTER_*：等的是哪个计数（0 = 没卡住）
  int sender;   // 多发送方里的第几个（机内到达用；其余 -1）
  int peer;  // 对端在本组里的组内序号（-1 = 所有发送方合计的一个格子）
  unsigned long long value, target;  // 当场读到的值 / 要等到的值
};
enum {
  COUNTER_INTRA_CREDIT = 1,  // 机内 credit
  COUNTER_INTRA_ARRIVE = 2,  // 机内到达（某个发送方那一格）
  COUNTER_INTER_CREDIT = 3,  // 机间 credit
  COUNTER_MOE_CREDIT = 4,    // 专家 credit
  COUNTER_INTER_SIGNAL = 5,  // 机间 signal（所有发送方合计）
  COUNTER_MOE_SIGNAL = 6     // 专家 signal（所有发送方合计）
};

// CE 能不能放：本机另外 ranks_per_node-1 张接收卡的机内 credit 格 [credit_intra
// + 8l] 都已达标
//   （它们都做完了占着这个槽的那个桶的机内求和，槽空了）。
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
// 机内求和能不能放：本桶的机内到达区里，本机另外每张发送卡那一格 [intra_arrive
// + (桶*16 + l)*8] 的序号
//   都已经追上本卡这个桶的 CE 次数（= 它们推过来的那几行都落地了）。
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
// dense put 能不能放：另外 num_nodes-1 个节点的机间 credit 格 [credit_inter +
// 8n] 都已达标（它们机间段的槽空了）。
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
// 专家 put 能不能放：专家组另外 expert_group_size-1 个 peer 的 credit 格
// [credit_moe + 8k] 都已达标。
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
// 最终求和能不能放：本桶的 signal 格已经涨到本桶发出的 put
// 笔数（别人的数据都到了本卡）。
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

// 记下出错位置并让本步停在这里（返回 -1 = 调用方往上传）
int record_failure(int rc, int stage, int bucket) {
  err_rc = rc;
  err_stage = stage;
  err_bucket = bucket;
  return -1;
}

// ---- Muon 的 2D 桶 gather：尾部线程放完一个 2D
// 桶的最终求和之后，紧接着把本卡这一片发给 owner。
//   2D 桶的每张卡只算出整桶里自己那一片，要由 owner
//   拼成完整的桶，所以最终求和之后还有一笔「发给 owner」。 表只由
//   setup_gather_table 建（只有 Muon 的 Python 侧会调）；没建表时
//   gather_enabled = 0，release_next_shard_reduce 不会走这个分支。
//   主线程（advance）放的最终求和不在这里发 put —— 它要等下一个 DeepEP
//   窗口，由 Python 侧的 GatherToOwner.window
//   发；所以这个分支只在尾部线程里走。 每个表项 = 一个 2D 桶：kind / 桶号
//   bucket / slot（选 event 用）/ 本卡是不是 owner / owner 在本组里的组内序号 /
//   本卡这一片在 send window 的偏移 send_offset 和元素数 shard_numel / owner
//   host window 上的到达格（跨节点用 arrive_offset，同节点用
//   intra_arrive_offset）。 dense 桶走 sharding 组、专家桶走 moe_sharding
//   组（各自的 comm / window / 机器分块）。 gather_seq[表项] = 本步这一片的 put
//   序号（同节点时 k_store_release 写进到达格的值），交接前由
//   set_gather_sequences 写入；0 = 本步不归尾部线程
//   发（主线程交接前已经放了它的最终求和）。 gather_sent / gather_sent_seq =
//   本步尾部线程发了哪些表项、用的什么序号，由 get_gather_sent 读出。
//   这些和其它每步状态一样，只在尾部线程不在 RUN 时由主线程读写。
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

// 一个桶的分片求和刚交到分片求和流（并记了 g_event_dense_inter_node_reduce /
// g_event_moe_reduce[bucket]）之后调。
//   不是 2D 桶、或者本卡就是 owner：什么都不做。
//   是发送方：gather put 流只等本卡这个 event（不用等窗口，尾部之后已经没有
//   DeepEP 了），然后 gather_put_to_owner（跨节点一笔 k_put_to_owner，同节点 CE
//   + k_store_release）。 不看远端条件（owner 的 send window
//   是整桶里固定的一段，没有槽要腾）。返回 0 或 CUDA 错误码。
int send_gather_put_after_shard_reduce(int kind, int bucket) {
  const int entry_index =
      (kind ? gather_index_moe : gather_index_dense)[bucket];
  if (entry_index < 0) return 0;
  const GatherEntry& entry = gather_entries[entry_index];
  if (entry.is_owner) return 0;
  const unsigned long long seq = gather_seq[entry_index];
  if (seq == 0)
    return (int)cudaErrorInvalidValue;  // 本步序号没给 / 本步已发过：不发第二次
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
  // 和 Python 侧一样核对发出的 signal 笔数：同节点走 CE 不发
  // signal（0），跨节点一笔 put 带一个 signal（1）
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

// 下面四个 release_next_*
// 各管一个阶段：从固定顺序里「还没放的下一个桶」开始，能放的都放，遇到第一个条件
//   不满足的就停（后面的桶顺序在它后面，也还不能放）。「这个桶在上一阶段的队列里」用位置比较表示：
//   某阶段的队列 = 它的位置和上一阶段位置之间那一段。
//   主线程（advance）和尾部线程调的都是这四个函数，差别只有传进来的
//   current_mark 和 block 数。 返回 1 = 这一轮放出去了东西，0 = 没有，-1 = CUDA
//   调用出错（已记进 err_*，本步停在这里）。
int release_next_intra_node_copy() {
  int moved = 0;
  while (pos_dense_copy < num_dense_buckets) {
    const int bucket = dense_release_order[pos_dense_copy];
    if (!ready_dense[bucket] || blocked_intra_node_copy(bucket, nullptr))
      break;  // 梯度凑齐了、而且接收卡的槽空了
    const DenseBucket& bucket_params = dense_buckets[bucket];
    // 本卡在各接收卡机内到达区里的那一格（每张接收卡上这一格的偏移相同）
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
    // 这个桶的 CE 已经放过（位置在 CE 之后），而且别人推来的行都到了
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
    // 这个桶的机内求和已经放过，而且对端机间段的槽空了
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
    dense_targets[bucket] +=
        (unsigned long long)num_signals;  // 本桶的 signal 目标值跟着涨
    ++stage_positions[1];
    moved = 1;
  }
  while (pos_moe_put < num_moe_buckets) {
    const int bucket = moe_release_order[pos_moe_put];
    if (!ready_moe[bucket] || blocked_moe_put(bucket, nullptr))
      break;  // 梯度凑齐了、而且 peer 的槽空了
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
// 最终求和这一阶段多两件事，所以多两个参数：
//   in_tail = 1（尾部线程）：放完一个 2D 桶的最终求和立刻发它的 gather
//   put（见上面那段注释）。 released
//   非空（主线程）：按放行顺序记下本次放了哪些最终求和（[kind, 桶号]
//   ...），Python 侧的
//     GatherToOwner 要用这个顺序登记、并在同一个窗口里发 2D gather put。
int release_next_shard_reduce(int in_tail, std::vector<int>* released) {
  int moved = 0;
  while (stage_positions[2] < num_dense_buckets) {
    const int bucket = dense_release_order[stage_positions[2]];
    // 这个桶的 put 已经放过，而且别人 put 过来的数据都到了
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
    if (in_tail && gather_enabled) {  // 只有 Muon 建了 2D 桶表才为 1
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
    if (in_tail && gather_enabled) {  // 同上（专家 2D 桶）
      current_activity = STAGE_GATHER_MOE * ACTIVITY_STAGE_SCALE + bucket + 1;
      const int gather_rc = send_gather_put_after_shard_reduce(1, bucket);
      current_activity = 0;
      if (gather_rc) return record_failure(gather_rc, STAGE_GATHER_MOE, bucket);
    }
    moved = 1;
  }
  return moved;
}
// 尾部线程的一轮：最终求和 → CE → 机内求和 → put
// 各试一轮。先试最终求和是因为它腾出接收池的槽，
//   这一轮后面的几个阶段马上就能用上。任一步出错立即返回 -1。
//   （主线程不走这个函数：它每个窗口只试 Python 指定的那几个阶段，见
//   advance。）
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
// 四个阶段「已经到这一阶段、还没放行」的桶数，顺序 = [CE, 机内求和, put,
// 最终求和]。
//   一个阶段的队列就是它的位置和上一阶段位置之间那一段：CE 和专家 put
//   这两头另要求桶已经 ready（它们的「上一阶段」是参数 hook 里的桶满）。
//   只有日志统计（Python 侧 summary 里的「尾部交给 proxy 多少 /
//   兜底多少」）用它。
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
// 本步放完了：六个位置都走到了头（四个阶段的队列都空）
bool is_step_done() {
  return pos_dense_copy >= num_dense_buckets &&
         stage_positions[0] >= num_dense_buckets &&
         stage_positions[1] >= num_dense_buckets &&
         stage_positions[2] >= num_dense_buckets &&
         pos_moe_put >= num_moe_buckets &&
         stage_positions[3] >= num_moe_buckets;
}

// 一步的主循环（RUN 期间，上面那些每步状态只有本线程碰）：取一条消息 → 反复
// release_pass_once 直到没有进展
//   → 放完就返回 DONE。既没有消息也没有进展时在条件变量上最多睡
//   POLL_INTERVAL_US（新消息 / all_posted / abort 都会把它叫醒）。CUDA
//   调用都在锁外做，不然主线程投消息会被堵住。
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
      // 本步「桶都交出来了」那一刻的现场：六个位置 +
      // 四个阶段还剩多少桶没放（日志用）
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
  // 本线程的定时器余量设成 1 µs（Linux 默认 50 µs：几 µs 的睡眠实际会被拉到 50
  // µs 以上）。只影响本线程
  prctl(PR_SET_TIMERSLACK, 1000UL, 0UL, 0UL, 0UL);
  std::unique_lock<std::mutex> lk(control->mutex);
  control->set_device_rc = (int)device_err;
  control->started = true;
  control->cv_main.notify_all();
  if (device_err != cudaSuccess) return;
  for (;;) {
    control->cv_proxy.wait(lk, [] {
      return control->state == RUN;
    });  // 步与步之间空闲等（不轮询）
    lk.unlock();
    const int state = run_step();
    lk.lock();
    control->state = state;
    control->cv_main.notify_all();
  }
}

}  // namespace

// 初始化（在 configure 之后调一次）：收下每桶参数表和固定放行顺序，建尾部
// mark 环，起尾部线程（起来就空闲等）。
//   桶数不传：从 g_event_dense_put / g_event_moe_put 的长度读（configure
//   按桶数开的那两组 event），所以不可能和引擎里的桶数对不上；传进来的固定顺序
//   和每桶参数表的长度要和它对得上，否则报错。
//   dense_table 每桶 7 个 long long、moe_table 每桶 5 个（字段见 DenseBucket /
//   MoeBucket）。 intra_arrive_offset_ 是 dense 组 host window
//   里机内到达区的字节偏移。 三块 credit 的位置不传：configure
//   已经收下过（g_credit_*_offset），这里直接换算成 u64 下标。
//   专家组大小 / 序号也不传：直接用 expert_group()（set_comms 存的那一份）。
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
  // 反过来建「桶号 → 位置」，顺带校验传进来的顺序是个排列（每个桶恰好出现一次）
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

// 主线程在计算流 stream 当前位置记一个尾部 mark，返回槽号；-1 = 没 setup /
// cudaEventRecord 失败， -2 = 本步的槽用完了（不绕回）
int record_tail_mark(cudaStream_t stream) {
  if (!control) return -1;
  if (tail_mark_index >= (int)tail_marks.size()) return -2;
  const int i = tail_mark_index;
  if (cudaEventRecord(tail_marks[i], stream) != cudaSuccess) return -1;
  ++tail_mark_index;
  return i;
}

// 每步开头（Python 侧 scheduler.end_step
// 给下一步调）：把每步状态清零，收下步号。
//   step 只影响 credit 目标（credit 是跨步累加的绝对值，见
//   credit_target），是 Python 侧那一份步号（engine.n_steps = 已结束的步数）的
//   绝对值，不在这里自己 ++ ：这一句万一被调两次，步号仍然是对的。
//   dense_copy_seq / dense_targets / moe_targets 是跨步累加的，这里**不**清零。
//   只在尾部线程不在 RUN 时可用（本步已经收尾过）。
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
  tail_mark_index = 0;  // 尾部 mark 环每步从 0 号槽重新开始
  return 0;
}

// 主线程在一个 DeepEP 窗口里放行：stages = 这个窗口要试哪些阶段（ADVANCE_*
// 的或，Python 侧定
//   哪个窗口试哪些），mark = 这个窗口的窗口 mark 槽号对应的 event，两个 block
//   数是这一轮交出去的 求和 kernel 用多少 block。选中的阶段按 CE → 机内求和 →
//   put → 最终求和 的顺序各试一轮
//   （每个阶段内部还是「能放的都放，遇到第一个不满足的就停」），不像尾部线程那样反复跑到没进展为止
//   —— 主线程不等远端，剩下的留给下一个窗口或尾部线程。
//   released_shard_reduce = 本次放了哪些最终求和（[kind, 桶号]
//   ...，按放行顺序），Muon 的 2D gather 要用。 返回 0 或 CUDA
//   错误码（出错位置在 err_*，用 get_last_error
//   取）。尾部线程已经接管时不许调（返回 cudaErrorInvalidValue）。
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

// 交接：本步最后一个 combine 之后，主线程把放行交给尾部线程，它进入 RUN。
//   状态（位置 / ready /
//   计数）已经在这里了，不用搬；只要给「尾部那两个阶段用多少 block」、
//   交接那一刻记的尾部 mark 槽号 mark_，以及 all_posted（= 1
//   表示本步的桶已经全部交出去了，兜底路径）。
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

// 某个桶的梯度凑齐了（kind 0 = dense 桶 bucket，1 = 专家桶
// bucket），或者尾部又过了一个
//   combine（kind -1，只换 mark）。scale = 这个桶最终求和之后要乘的系数。
//   交接之前（IDLE）：mark 必须是 -1，这里直接把 ready / scale
//   记下来，等下一个窗口的 advance 放它。 交接之后（RUN）：mark 是尾部 mark
//   的槽号，投一条消息给尾部线程。 返回 0 = 收下了；1 =
//   尾部线程已经出错停下（用 get_last_error 取原因）； 2 = 用法不对（mark
//   和状态不配 / 桶号无效 / 这个桶本步已经 ready 过 / 全部放完后又来了一个
//   ready 的桶）
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
        (kind ? ready_moe : ready_dense)[bucket];  // 本步只能 ready 一次
    if (ready) return 2;
    ready = 1;
    (kind ? moe_scales : dense_scales)[bucket] = scale;
    return 0;
  }
  if (mark_ < 0 || mark_ >= tail_mark_index) return 2;
  if (control->state == DONE)
    return kind < 0 ? 0 : 2;  // 已全部放完：只换 mark 的消息没有用处，直接丢
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

// 等尾部线程离开 RUN（放完 / 出错 / 被叫停）或者超时（超时返回时状态仍是
// RUN），返回当时的状态。
//   条件变量等待，不轮询。
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

// 叫停本步（超时报错前用）：尾部线程在每轮开头看 abort 标志；它要是卡在某个
// CUDA 调用里，这里会等超时、仍返回 RUN
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

// 尾部线程停下之后收尾，回到 IDLE（只在它已离开 RUN 时可用）：清掉消息队列和
// all_posted / abort 两个标志，
//   主线程重新成为唯一写者（放到哪儿了、还剩什么，用 status / stuck_report
//   查）。
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

// 本步放行到哪儿了（主线程查，用于日志和「本步每个桶都放行了吗」的检查）：
//   positions6 = [CE, dense 机内求和, dense put, dense 最终求和, 专家 put,
//   专家最终求和]， counts4 = 四个阶段还有多少桶没放（顺序同
//   pending_counts），posted4 = 尾部线程看到
//   「桶都交出来了」那一刻的同一组数（没看到过那一刻时是
//   [-1,0,0,0]），*done = 六个位置是不是都到头了。
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

// 超时现场（主线程查，只读 host
// window，不改任何状态）：六个阶段各一条记录，顺序 =
//   报告里打印的顺序（dense 的四个阶段，再专家的两个）。一条记录是
//     [阶段编号 STAGE_*, 位置, 桶数, 下一个桶到没到这一阶段（1 = 到了，0 =
//     还没）, 还没放行的桶数, 下面列了几个]
//   后面跟「列了几个」× 6 个数（每个桶一组，最多 STUCK_MAX_ITEMS 个）：
//     [桶号, 等的是哪个计数 COUNTER_*（0 =
//     条件已满足，排在固定顺序前一个后面）, 多发送方里的第几个（-1 = 不适用）,
//     对端组内序号（-1 = 所有发送方合计）, 读到的值, 目标值]
//   Python 侧（scheduler._stuck）把它拼成人看的一行行文字。
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
  // 这个阶段的队列：位置之后、已经到了这一阶段的那些桶（到没到 = ready
  // 或者上一阶段的位置在它后面）
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

// 此刻正在发哪个调用：阶段编号 * ACTIVITY_STAGE_SCALE + 桶号 + 1；0 = 不在 CUDA
// 调用里（超时报告用）
int get_current_activity() { return current_activity.load(); }

// 导出给 Python 的常量表：名字 →
// 数值。这些数值只在这个文件里定义一份（上面那几个
//   enum 和 ACTIVITY_STAGE_SCALE）—— advance 的阶段集合 ADVANCE_*、阶段编号
//   STAGE_*、超时报告里的计数编号 COUNTER_*、尾部线程状态 TAIL_*、
//   get_current_activity 的编码倍率。gin_overlap_api.cc 把它们导出成
//   paddle.base.core.gin_overlap 的模块属性，Python
//   侧
//   按名字取（scheduler._bind_constants）而不是自己再写一遍数值：改一边忘另一边
//   不会崩、只会把阶段放错窗口或者把现场译错，是最难发现的那类 bug。
//   加 / 改 / 删 STAGE_* / COUNTER_* 时这张表要跟着改，Python
//   侧的译名表对不上会报错。
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

// 建 2D 桶表（在 setup 和 gather_setup 之后、尾部线程处于 IDLE 时调）。table
// 每项 10 个 long long：
//   [kind（0 dense / 1 专家）, 桶号, slot, 本卡是不是 owner, owner 的组内序号,
//   send_offset, shard_numel,
//    保留位（必须 ≥ 1，历史上是分片数，现在恒当 1 用）, arrive_offset,
//    intra_arrive_offset]。
//   表为空时不建，gather_enabled 保持 0。
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
      return (int)cudaErrorInvalidValue;  // 同一个桶给了两项
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

// 主线程在一个 DeepEP 窗口里发一个 2D 桶的 gather put：参数不再从 Python
// 逐个传过来，按表项下标（= Python 侧 gather_keys 的次序，和 setup_gather_table
// 收下的那张表同一份）查表。等的是门流上记的
// g_event_gather_source[slot]（主线程已经让门流等过窗口 mark 和本卡本桶那三个
// event）；尾部线程那条路在 send_gather_put_after_shard_reduce。
//   seq = 本步这一片的 put 序号（同节点时 k_store_release 写进 owner
//   的到达格）， 0 不合法。*num_signals 回给 Python 核对笔数（同节点 0 / 跨节点
//   1）。 只许主线程在尾部线程不接管时调（状态 IDLE）。
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

// 交接之前（tail_start 之前，尾部线程还在 IDLE）：给出本步每个表项的 put
// 序号（0 = 本步不归它发），
//   并清掉上一步的发送记录
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

// 尾部线程已离开 RUN 之后：flat = 本步它发的每个 gather put [kind, bucket,
// seq]（按发出顺序）
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
