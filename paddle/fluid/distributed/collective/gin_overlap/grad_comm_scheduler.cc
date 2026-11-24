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

#include "paddle/fluid/distributed/collective/gin_overlap/grad_comm_scheduler.h"

#include <sys/prctl.h>

#include <algorithm>
#include <chrono>
#include <sstream>

#include "glog/logging.h"
#include "paddle/common/errors.h"
#include "paddle/fluid/distributed/collective/gin_overlap/kernels/sum.h"
#include "paddle/fluid/distributed/collective/process_group_nccl.h"
#include "paddle/phi/api/include/context_pool.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/cuda_stream.h"
#include "paddle/phi/core/distributed/nccl_comm_context.h"
#include "paddle/phi/core/enforce.h"

namespace paddle {
namespace distributed {
namespace {

// States of the scheduler. kIdle: the caller of Release() releases (also
// between steps). kRun: the proxy thread releases. kDone / kErr / kAborted:
// the proxy thread released the whole step / stopped on an error / stopped on
// request.
enum State { kIdle = 0, kRun = 1, kDone = 2, kErr = 3, kAborted = 4 };

// Which counter a blocked op waits for (Blocker::counter, StuckReport()).
enum Counter {
  kCounterIntraCredit = 1,
  kCounterIntraArrive = 2,
  kCounterInterCredit = 3,
  kCounterExpertCredit = 4,
  kCounterInterSignal = 5,
  kCounterExpertSignal = 6,
};

constexpr int kAllOps = kGinOpIntraCopy | kGinOpIntraSum | kGinOpPut |
                        kGinOpFinalSum | kGinOpGather;
// Longest wait of the proxy thread on its condition variable when it has
// neither progress nor a new report. Remote counters do not wake it, so they
// are polled.
constexpr int kPollIntervalUs = 10;
// Sleep of the owner between two polls without progress in FinishStep().
constexpr int kGatherPollUs = 50;
// How long FinishStep() waits for the proxy thread to stop after a timeout.
constexpr double kAbortWaitSeconds = 10.0;
// Sum kernel grid sizes when FinishStep() hands over to the proxy thread
// itself (the step never called ReleaseAsync()): the counts of the pre
// window, not those of the tail.
constexpr int kFallbackIntraSumBlocks = 16;
constexpr int kFallbackFinalSumBlocks = 64;
// activity_ = op * kActivityScale + bucket + 1.
constexpr int kActivityScale = 1000000;
// slot_bucket_ entries that are not a bucket id.
constexpr int kSlotUnused = -2;
constexpr int kSlotMarkOnly = -1;
// Intra-node arrival counters per dense bucket (ranks_per_node <= 16).
constexpr int kMaxRanksPerNode = 16;
// Buckets listed per op in StuckReport().
constexpr int kStuckMaxItems = 8;
constexpr int64_t kCounterBytes = 8;
constexpr int64_t kFloatBytes = 4;

const char* OpName(int op) {
  switch (op) {
    case kGinOpIntraCopy:
      return "INTRA_COPY";
    case kGinOpIntraSum:
      return "INTRA_SUM";
    case kGinOpPut:
      return "PUT";
    case kGinOpFinalSum:
      return "FINAL_SUM";
    case kGinOpGather:
      return "GATHER";
    default:
      return "?";
  }
}

const char* CounterName(int counter) {
  switch (counter) {
    case kCounterIntraCredit:
      return "intra-node credit";
    case kCounterIntraArrive:
      return "intra-node arrival";
    case kCounterInterCredit:
      return "inter-node credit";
    case kCounterExpertCredit:
      return "expert credit";
    case kCounterInterSignal:
      return "inter-node signal";
    case kCounterExpertSignal:
      return "expert signal";
    default:
      return "?";
  }
}

// Paddle's current compute stream of the device, the stream of
// paddle.device.current_stream().
cudaStream_t ComputeStream(int device_id) {
  return paddle::GetCurrentCUDAStream(phi::GPUPlace(device_id))->raw_stream();
}

cudaStream_t CreateStream() {
  cudaStream_t stream = nullptr;
  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  return stream;
}

void CreateEvents(size_t count, std::vector<cudaEvent_t>* events) {
  events->assign(count, nullptr);
  for (cudaEvent_t& event : *events) {
    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
  }
}

// Whether [offset, offset + count * unit) lies in a window of `bytes` bytes
// and offset is a multiple of `align`. Nothing can wrap around.
bool InWindow(
    int64_t offset, int64_t count, int64_t unit, int64_t align, size_t bytes) {
  if (offset < 0 || count < 0 || offset % align != 0) return false;
  if (static_cast<size_t>(offset) > bytes) return false;
  return static_cast<size_t>(count) <=
         (bytes - static_cast<size_t>(offset)) / static_cast<size_t>(unit);
}

}  // namespace

GinGradCommScheduler::GinGradCommScheduler(
    const GinGradCommSchedulerConfig& config) {
  PADDLE_ENFORCE_GPU_SUCCESS(cudaGetDevice(&device_id_));
  const int num_buckets = static_cast<int>(config.buckets.size());
  int num_dense = 0;
  int num_expert = 0;
  for (const GinBucketDesc& bucket_desc : config.buckets) {
    if (bucket_desc.kind == kGinDenseBucket) {
      ++num_dense;
    } else {
      ++num_expert;
    }
  }
  PADDLE_ENFORCE_EQ(num_expert == 0 || config.expert_group != nullptr,
                    true,
                    common::errors::InvalidArgument(
                        "The bucket table has %d expert buckets but no "
                        "expert process group (moe_sharding group).",
                        num_expert));
  InitGroup(kGinDenseBucket, config.dense_group, config.dense_windows);
  if (num_expert > 0) {
    InitGroup(kGinExpertBucket, config.expert_group, config.expert_windows);
  }
  buckets_ = config.buckets;
  dense_order_ = config.dense_order;
  expert_order_ = config.expert_order;
  credit_intra_offset_ = config.credit_intra_offset;
  credit_inter_offset_ = config.credit_inter_offset;
  credit_expert_offset_ = config.credit_expert_offset;

  // Each order must be a permutation of the bucket ids of its kind.
  position_.assign(num_buckets, -1);
  auto set_positions = [&](const std::vector<int>& order,
                           int kind,
                           int count,
                           const char* name) {
    bool is_permutation = static_cast<int>(order.size()) == count;
    for (int position = 0; is_permutation && position < count; ++position) {
      const int bucket = order[position];
      is_permutation = bucket >= 0 && bucket < num_buckets &&
                       buckets_[bucket].kind == kind && position_[bucket] < 0;
      if (is_permutation) position_[bucket] = position;
    }
    PADDLE_ENFORCE_EQ(is_permutation,
                      true,
                      common::errors::InvalidArgument(
                          "The %s release order is not a permutation of the "
                          "ids of the %d %s buckets.",
                          name,
                          count,
                          name));
  };
  set_positions(dense_order_, kGinDenseBucket, num_dense, "dense");
  set_positions(expert_order_, kGinExpertBucket, num_expert, "expert");
  CheckTables(config);

  gather_index_.assign(num_buckets, -1);
  for (size_t entry_index = 0; entry_index < config.gather.size();
       ++entry_index) {
    const GinGatherDesc& gather_desc = config.gather[entry_index];
    const bool is_new_bucket = gather_desc.bucket >= 0 &&
                               gather_desc.bucket < num_buckets &&
                               gather_index_[gather_desc.bucket] < 0;
    PADDLE_ENFORCE_EQ(
        is_new_bucket && gather_desc.owner >= 0 &&
            gather_desc.owner < GroupOf(gather_desc.bucket).size &&
            InWindow(gather_desc.arrive_offset,
                     1,
                     kCounterBytes,
                     kCounterBytes,
                     GroupOf(gather_desc.bucket).host_bytes) &&
            InWindow(gather_desc.intra_arrive_offset,
                     kMaxRanksPerNode,
                     kCounterBytes,
                     kCounterBytes,
                     GroupOf(gather_desc.bucket).host_bytes),
        true,
        common::errors::InvalidArgument(
            "Gather entry %d (bucket %d, owner %d): the bucket is out of "
            "range or already has an entry, the owner is not a rank of its "
            "group, or its arrival counters lie outside the host window.",
            static_cast<int>(entry_index),
            gather_desc.bucket,
            gather_desc.owner));
    const Group& group = GroupOf(gather_desc.bucket);
    GatherEntry entry;
    entry.gather_desc = gather_desc;
    entry.is_owner = gather_desc.owner == group.rank;
    if (entry.is_owner) {
      for (int peer = 0; peer < group.size; ++peer) {
        if (peer == group.rank) continue;
        if (peer / group.ranks_per_node != group.node_index) {
          ++entry.num_inter_senders;
        } else {
          entry.intra_senders.push_back(peer % group.ranks_per_node);
        }
      }
    }
    gather_index_[gather_desc.bucket] = static_cast<int>(entry_index);
    gather_.push_back(entry);
  }

  ready_.assign(num_buckets, 0);
  scale_.assign(num_buckets, 1.f);
  reported_.assign(num_buckets, 0);
  copy_seq_.assign(num_buckets, 0);
  targets_.assign(num_buckets, 0);
  published_scale_.assign(num_buckets, 1.f);
  gather_owner_step_.assign(gather_.size(), 0);
  // The destructor does not run when the constructor raises, so the streams
  // and events are destroyed here on that path.
  try {
    CreateStreamsAndEvents(config.num_window_marks, config.num_tail_marks);
    slot_bucket_.assign(tail_marks_.size(), kSlotUnused);

    proxy_ = std::thread([this] { ProxyMain(); });
    std::unique_lock<std::mutex> lock(mutex_);
    cv_main_.wait(lock, [this] { return proxy_started_; });
    if (proxy_device_status_ != 0) {
      const int status = proxy_device_status_;
      lock.unlock();
      proxy_.join();
      PADDLE_THROW(common::errors::External(
          "cudaSetDevice(%d) failed in the proxy thread of "
          "GinGradCommScheduler: %s.",
          device_id_,
          cudaGetErrorString(static_cast<cudaError_t>(status))));
    }
  } catch (...) {
    DestroyStreamsAndEvents();
    throw;
  }
}

GinGradCommScheduler::~GinGradCommScheduler() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stop_ = true;
    abort_ = true;
    cv_proxy_.notify_all();
  }
  if (proxy_.joinable()) proxy_.join();
  DestroyStreamsAndEvents();
}

void GinGradCommScheduler::DestroyStreamsAndEvents() {
  for (cudaStream_t stream : {stream_intra_copy_,
                              stream_intra_sum_,
                              stream_put_,
                              stream_final_sum_,
                              stream_gather_put_,
                              stream_gather_gate_}) {
    if (stream) cudaStreamDestroy(stream);
  }
  for (std::vector<cudaEvent_t>* events : {&event_intra_copy_,
                                           &event_intra_sum_,
                                           &event_put_,
                                           &event_final_sum_,
                                           &event_gather_source_,
                                           &event_gather_put_,
                                           &window_marks_,
                                           &tail_marks_}) {
    for (cudaEvent_t event : *events) {
      if (event) cudaEventDestroy(event);
    }
  }
}

void GinGradCommScheduler::InitGroup(int kind,
                                     ProcessGroupNCCL* pg,
                                     const GinGroupWindows& windows) {
  const char* name = kind == kGinDenseBucket ? "sharding" : "moe_sharding";
  Group& group = groups_[kind];
  group.pg = pg;
  group.windows = windows;
  // Raises when the device communicator of the group does not exist.
  const phi::distributed::NCCLDevCommInfo dev_comm_info = pg->GetDevCommInfo();
  phi::distributed::NCCLCommContext* comm_context =
      pg->GetOrCreateCommContext(phi::GPUPlace(device_id_));
  // A window not registered on the communicator keeps its size at 0, so
  // CheckTables() rejects every range in it.
  auto lookup = [&](void* window, size_t* bytes) {
    void* window_base = nullptr;
    comm_context->GetWindowRange(window, &window_base, bytes);
    return static_cast<char*>(window_base);
  };
  group.send = lookup(windows.send_window, &group.send_bytes);
  group.recv = lookup(windows.recv_window, &group.recv_bytes);
  group.host = lookup(windows.host_window, &group.host_bytes);
  const int lsa_size = dev_comm_info.lsa_size;
  PADDLE_ENFORCE_EQ(
      lsa_size >= 1 && lsa_size <= kMaxRanksPerNode &&
          dev_comm_info.nranks % lsa_size == 0 &&
          dev_comm_info.lsa_rank == dev_comm_info.rank % lsa_size,
      true,
      common::errors::PreconditionNotMet(
          "The %s group needs node-major ranks and at most %d ranks per LSA "
          "team, but rank %d of %d has LSA rank %d in a team of %d.",
          name,
          kMaxRanksPerNode,
          dev_comm_info.rank,
          dev_comm_info.nranks,
          dev_comm_info.lsa_rank,
          lsa_size));
  group.size = dev_comm_info.nranks;
  group.rank = dev_comm_info.rank;
  group.ranks_per_node = lsa_size;
  group.rank_in_node = dev_comm_info.lsa_rank;
  group.num_nodes = dev_comm_info.nranks / lsa_size;
  group.node_index = dev_comm_info.rank / lsa_size;
}

void GinGradCommScheduler::CheckTables(
    const GinGradCommSchedulerConfig& config) const {
  const Group& dense = groups_[kGinDenseBucket];
  const Group& expert = groups_[kGinExpertBucket];
  const int64_t ranks_per_node = dense.ranks_per_node;
  const int64_t num_nodes = dense.num_nodes;
  // the first range that lies outside its window or is misaligned
  std::ostringstream bad_range;
  if (!InWindow(credit_intra_offset_,
                kMaxRanksPerNode,
                kCounterBytes,
                kCounterBytes,
                dense.host_bytes) ||
      !InWindow(credit_inter_offset_,
                num_nodes,
                kCounterBytes,
                kCounterBytes,
                dense.host_bytes)) {
    bad_range << "The intra-node / inter-node credit area (sharding group host "
                 "window of "
              << dense.host_bytes << " bytes)";
  } else if (!config.expert_order.empty() && !InWindow(credit_expert_offset_,
                                                       expert.size,
                                                       kCounterBytes,
                                                       kCounterBytes,
                                                       expert.host_bytes)) {
    bad_range << "The expert credit area (moe_sharding group host window of "
              << expert.host_bytes << " bytes)";
  }
  for (size_t bucket = 0;
       bad_range.tellp() == 0 && bucket < config.buckets.size();
       ++bucket) {
    const GinBucketDesc& bucket_desc = config.buckets[bucket];
    const bool is_dense = bucket_desc.kind == kGinDenseBucket;
    const Group& group = is_dense ? dense : expert;
    const int64_t numel = bucket_desc.shard_numel;
    // The intra-node sum reads the dense send and intra-node recv ranges as
    // float4, so those are 16-byte aligned and a dense shard is a multiple
    // of 4.
    bool is_valid_layout = numel > 0 && (!is_dense || numel % 4 == 0) &&
                           numel <= INT64_MAX / kFloatBytes / group.size;
    is_valid_layout = is_valid_layout && InWindow(bucket_desc.send_offset,
                                                  numel * group.size,
                                                  kFloatBytes,
                                                  is_dense ? 16 : kFloatBytes,
                                                  group.send_bytes);
    is_valid_layout = is_valid_layout && InWindow(bucket_desc.signal_offset,
                                                  1,
                                                  kCounterBytes,
                                                  kCounterBytes,
                                                  group.host_bytes);
    if (is_dense) {
      is_valid_layout = is_valid_layout && InWindow(bucket_desc.arrive_offset,
                                                    kMaxRanksPerNode,
                                                    kCounterBytes,
                                                    kCounterBytes,
                                                    group.host_bytes);
      is_valid_layout =
          is_valid_layout && (bucket_desc.depth[0] < 0 ||
                              InWindow(bucket_desc.recv_offset[0],
                                       (ranks_per_node - 1) * num_nodes * numel,
                                       kFloatBytes,
                                       16,
                                       group.recv_bytes));
      is_valid_layout = is_valid_layout && (bucket_desc.depth[1] < 0 ||
                                            InWindow(bucket_desc.recv_offset[1],
                                                     num_nodes * numel,
                                                     kFloatBytes,
                                                     kFloatBytes,
                                                     group.recv_bytes));
    } else {
      is_valid_layout = is_valid_layout && (bucket_desc.depth[0] < 0 ||
                                            InWindow(bucket_desc.recv_offset[0],
                                                     group.size * numel,
                                                     kFloatBytes,
                                                     kFloatBytes,
                                                     group.recv_bytes));
    }
    if (!is_valid_layout) {
      bad_range << "Bucket " << bucket << " ("
                << (is_dense ? "dense" : "expert") << ", shard of " << numel
                << " elements): the shard size, or one of its ranges (send, "
                   "signal, arrival counters, recv slots)";
    }
  }
  PADDLE_ENFORCE_EQ(
      bad_range.tellp() == 0,
      true,
      common::errors::InvalidArgument(
          "%s lies outside its window or is misaligned.", bad_range.str()));
}

void GinGradCommScheduler::CreateStreamsAndEvents(int num_window_marks,
                                                  int num_tail_marks) {
  stream_intra_copy_ = CreateStream();
  stream_intra_sum_ = CreateStream();
  stream_put_ = CreateStream();
  stream_final_sum_ = CreateStream();
  stream_gather_put_ = CreateStream();
  // Only cudaStreamWaitEvent and cudaEventRecord go on the gate stream.
  stream_gather_gate_ = CreateStream();
  const size_t num_buckets = buckets_.size();
  CreateEvents(num_buckets, &event_intra_copy_);
  CreateEvents(num_buckets, &event_intra_sum_);
  CreateEvents(num_buckets, &event_put_);
  CreateEvents(num_buckets, &event_final_sum_);
  CreateEvents(gather_.size(), &event_gather_source_);
  CreateEvents(gather_.size(), &event_gather_put_);
  CreateEvents(num_window_marks, &window_marks_);
  CreateEvents(num_tail_marks, &tail_marks_);
}

const GinGradCommScheduler::Group& GinGradCommScheduler::GroupOf(
    int bucket) const {
  return groups_[buckets_[bucket].kind];
}

// Acquire load of a u64 counter of this rank's host window, always from
// memory.
uint64_t GinGradCommScheduler::HostCounter(const Group& group,
                                           int64_t byte_offset) const {
  return __atomic_load_n(
      reinterpret_cast<const uint64_t*>(group.host + byte_offset),
      __ATOMIC_ACQUIRE);
}

// Credit target of a recv pool slot. The bucket at `position` of its fixed
// order in step `step_` has sequence number step_ * num_buckets + position;
// the previous holder of its slot has a sequence number smaller by `depth`.
// The peer returns one credit per bucket it is done with, and credits
// accumulate across steps, so target = step_ * num_buckets + position - depth
// + 1 (0 if negative). depth < 0: no slot, nothing to wait for.
uint64_t GinGradCommScheduler::CreditTarget(int num_buckets,
                                            int position,
                                            int64_t depth) const {
  if (depth < 0) return 0;
  const int64_t target = step_ * num_buckets + position - depth + 1;
  return target > 0 ? static_cast<uint64_t>(target) : 0;
}

// The Blocked* functions are the remote release conditions; they only read
// counters of this rank's host windows. true = cannot be released yet;
// `blocker` (may be null) receives the counter waited for.

// INTRA_COPY: the intra-node credits from every other rank of the node have
// reached the target, i.e. this bucket's slot in their intra-node segments is
// free.
bool GinGradCommScheduler::BlockedIntraCopy(int bucket,
                                            Blocker* blocker) const {
  const Group& group = groups_[kGinDenseBucket];
  const uint64_t target = CreditTarget(static_cast<int>(dense_order_.size()),
                                       position_[bucket],
                                       buckets_[bucket].depth[0]);
  for (int local_rank = 0; local_rank < group.ranks_per_node; ++local_rank) {
    if (local_rank == group.rank_in_node) continue;
    const uint64_t value =
        HostCounter(group, credit_intra_offset_ + kCounterBytes * local_rank);
    if (value < target) {
      if (blocker) {
        *blocker = Blocker{kCounterIntraCredit,
                           -1,
                           group.node_index * group.ranks_per_node + local_rank,
                           value,
                           target};
      }
      return true;
    }
  }
  return false;
}

// INTRA_SUM: in this bucket's intra-node arrival counters, the counter of
// every other sender of the node has reached this rank's INTRA_COPY count of
// the bucket. The counter of node-local rank `local_rank` has one sender,
// that rank, which adds 1 to it
// per INTRA_COPY of the bucket, i.e. once per step, like this rank's own
// count copy_seq_[bucket]; so after the INTRA_COPY of step s the target is s.
bool GinGradCommScheduler::BlockedIntraSum(int bucket, Blocker* blocker) const {
  const Group& group = groups_[kGinDenseBucket];
  const uint64_t target = copy_seq_[bucket];
  for (int local_rank = 0; local_rank < group.ranks_per_node; ++local_rank) {
    if (local_rank == group.rank_in_node) continue;
    const uint64_t value = HostCounter(
        group, buckets_[bucket].arrive_offset + kCounterBytes * local_rank);
    if (value < target) {
      if (blocker) {
        *blocker = Blocker{kCounterIntraArrive,
                           local_rank,
                           group.node_index * group.ranks_per_node + local_rank,
                           value,
                           target};
      }
      return true;
    }
  }
  return false;
}

// Dense PUT: the inter-node credits from every other node have reached the
// target.
bool GinGradCommScheduler::BlockedDensePut(int bucket, Blocker* blocker) const {
  const Group& group = groups_[kGinDenseBucket];
  const uint64_t target = CreditTarget(static_cast<int>(dense_order_.size()),
                                       position_[bucket],
                                       buckets_[bucket].depth[1]);
  for (int node = 0; node < group.num_nodes; ++node) {
    if (node == group.node_index) continue;
    const uint64_t value =
        HostCounter(group, credit_inter_offset_ + kCounterBytes * node);
    if (value < target) {
      if (blocker) {
        *blocker = Blocker{kCounterInterCredit,
                           -1,
                           node * group.ranks_per_node + group.rank_in_node,
                           value,
                           target};
      }
      return true;
    }
  }
  return false;
}

// Expert PUT: the expert credits from every other rank of the expert group
// have reached the target.
bool GinGradCommScheduler::BlockedExpertPut(int bucket,
                                            Blocker* blocker) const {
  const Group& group = groups_[kGinExpertBucket];
  const uint64_t target = CreditTarget(static_cast<int>(expert_order_.size()),
                                       position_[bucket],
                                       buckets_[bucket].depth[0]);
  for (int group_rank = 0; group_rank < group.size; ++group_rank) {
    if (group_rank == group.rank) continue;
    const uint64_t value =
        HostCounter(group, credit_expert_offset_ + kCounterBytes * group_rank);
    if (value < target) {
      if (blocker)
        *blocker = Blocker{kCounterExpertCredit, -1, group_rank, value, target};
      return true;
    }
  }
  return false;
}

// FINAL_SUM: the bucket's signal counter has reached the number of puts this
// rank issued for it (the ranks are symmetric), i.e. the puts of all other
// ranks have arrived.
bool GinGradCommScheduler::BlockedDenseFinalSum(int bucket,
                                                Blocker* blocker) const {
  const uint64_t value =
      HostCounter(groups_[kGinDenseBucket], buckets_[bucket].signal_offset);
  if (value >= targets_[bucket]) return false;
  if (blocker) {
    *blocker = Blocker{kCounterInterSignal, -1, -1, value, targets_[bucket]};
  }
  return true;
}

bool GinGradCommScheduler::BlockedExpertFinalSum(int bucket,
                                                 Blocker* blocker) const {
  const uint64_t value =
      HostCounter(groups_[kGinExpertBucket], buckets_[bucket].signal_offset);
  if (value >= targets_[bucket]) return false;
  if (blocker) {
    *blocker = Blocker{kCounterExpertSignal, -1, -1, value, targets_[bucket]};
  }
  return true;
}

// The Submit* functions submit one op of one bucket; the remote conditions
// were checked before. Work waits only on current_mark_ and local events.

// INTRA_COPY (stream_intra_copy_): wait on the mark -> this rank's blocks of
// column `local_rank` go to row `row` of the intra-node segment of node-local
// rank `local_rank` -> this rank's arrival counter of the bucket on
// `local_rank` is incremented by 1 (after
// the copies, same stream) -> record the event. Nothing is submitted with one
// rank per node.
void GinGradCommScheduler::SubmitIntraCopy(int bucket) {
  const Group& group = groups_[kGinDenseBucket];
  const GinBucketDesc& bucket_desc = buckets_[bucket];
  const int ranks_per_node = group.ranks_per_node;
  const int num_nodes = group.num_nodes;
  const int rank_in_node = group.rank_in_node;
  if (ranks_per_node < 2) return;
  const int64_t block_bytes = bucket_desc.shard_numel * kFloatBytes;
  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaStreamWaitEvent(stream_intra_copy_, current_mark_, 0));
  std::vector<CopyToPeersArgs> copies;
  std::vector<IncrementPeerCounterArgs> arrivals;
  copies.reserve(static_cast<size_t>(ranks_per_node - 1) * num_nodes);
  arrivals.reserve(static_cast<size_t>(ranks_per_node - 1));
  for (int local_rank = 0; local_rank < ranks_per_node; ++local_rank) {
    if (local_rank == rank_in_node) continue;
    const int peer = group.node_index * ranks_per_node + local_rank;
    // this rank's row on receiver local_rank
    const int row = local_rank > rank_in_node ? rank_in_node : rank_in_node - 1;
    for (int node = 0; node < num_nodes; ++node) {
      CopyToPeersArgs copy;
      copy.src_window = group.windows.send_window;
      copy.src_offset = static_cast<size_t>(
          bucket_desc.send_offset +
          (static_cast<int64_t>(node) * ranks_per_node + local_rank) *
              block_bytes);
      copy.dst_window = group.windows.recv_window;
      copy.dst_offset = static_cast<size_t>(
          bucket_desc.recv_offset[0] +
          (static_cast<int64_t>(row) * num_nodes + node) * block_bytes);
      copy.peer = peer;
      copy.bytes = static_cast<size_t>(block_bytes);
      copies.push_back(copy);
    }
    IncrementPeerCounterArgs arrival;
    arrival.peer = peer;
    arrival.counter_window = group.windows.host_window;
    arrival.counter_offset = static_cast<size_t>(bucket_desc.arrive_offset +
                                                 kCounterBytes * rank_in_node);
    arrivals.push_back(arrival);
  }
  group.pg->CopyToPeers(copies, stream_intra_copy_);
  group.pg->IncrementPeerCounter(arrivals, stream_intra_copy_);
  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaEventRecord(event_intra_copy_[bucket], stream_intra_copy_));
}

// INTRA_SUM (stream_intra_sum_): wait on the mark -> each block of this
// rank's column becomes the sum over the node, in place -> an intra-node
// credit goes to every other rank of the node -> record the event.
void GinGradCommScheduler::SubmitIntraSum(int bucket) {
  const Group& group = groups_[kGinDenseBucket];
  const GinBucketDesc& bucket_desc = buckets_[bucket];
  const int ranks_per_node = group.ranks_per_node;
  const int rank_in_node = group.rank_in_node;
  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaStreamWaitEvent(stream_intra_sum_, current_mark_, 0));
  PADDLE_ENFORCE_GPU_SUCCESS(LaunchIntraNodeSum(
      reinterpret_cast<float*>(group.send + bucket_desc.send_offset),
      reinterpret_cast<const float*>(group.recv + bucket_desc.recv_offset[0]),
      static_cast<size_t>(bucket_desc.shard_numel),
      ranks_per_node,
      group.num_nodes,
      rank_in_node,
      intra_sum_blocks_,
      stream_intra_sum_));
  if (ranks_per_node > 1) {
    std::vector<IncrementPeerCounterArgs> credits;
    credits.reserve(static_cast<size_t>(ranks_per_node - 1));
    for (int local_rank = 0; local_rank < ranks_per_node; ++local_rank) {
      if (local_rank == rank_in_node) continue;
      IncrementPeerCounterArgs credit;
      credit.peer = group.node_index * ranks_per_node + local_rank;
      credit.counter_window = group.windows.host_window;
      credit.counter_offset = static_cast<size_t>(credit_intra_offset_ +
                                                  kCounterBytes * rank_in_node);
      credits.push_back(credit);
    }
    group.pg->IncrementPeerCounter(credits, stream_intra_sum_);
  }
  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaEventRecord(event_intra_sum_[bucket], stream_intra_sum_));
}

// Dense PUT (stream_put_): wait on the bucket's INTRA_SUM event and on the
// mark -> block peer_node of this column goes to the rank with the same
// rank_in_node on node peer_node, into row node_index of its inter-node
// segment, incrementing its signal counter of the bucket -> record the event.
// Returns how much the peers' signal counters grow.
int GinGradCommScheduler::SubmitDensePut(int bucket) {
  const Group& group = groups_[kGinDenseBucket];
  const GinBucketDesc& bucket_desc = buckets_[bucket];
  const int ranks_per_node = group.ranks_per_node;
  const int num_nodes = group.num_nodes;
  const int64_t block_bytes = bucket_desc.shard_numel * kFloatBytes;
  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaStreamWaitEvent(stream_put_, event_intra_sum_[bucket], 0));
  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaStreamWaitEvent(stream_put_, current_mark_, 0));
  int num_signals = 0;
  if (num_nodes > 1) {
    std::vector<PutArgs> puts;
    puts.reserve(static_cast<size_t>(num_nodes - 1));
    for (int peer_index = 0; peer_index < num_nodes - 1; ++peer_index) {
      const int peer_node = (group.node_index + 1 + peer_index) % num_nodes;
      PutArgs put;
      put.src_window = group.windows.send_window;
      put.src_offset = static_cast<size_t>(
          bucket_desc.send_offset +
          (static_cast<int64_t>(peer_node) * ranks_per_node +
           group.rank_in_node) *
              block_bytes);
      put.dst_window = group.windows.recv_window;
      put.dst_offset = static_cast<size_t>(bucket_desc.recv_offset[1] +
                                           group.node_index * block_bytes);
      put.bytes = static_cast<size_t>(block_bytes);
      put.peer = peer_node * ranks_per_node + group.rank_in_node;
      put.counter_window = group.windows.host_window;
      put.counter_offset = static_cast<size_t>(bucket_desc.signal_offset);
      puts.push_back(put);
    }
    group.pg->Put(puts, stream_put_);
    num_signals = num_nodes - 1;
  }
  PADDLE_ENFORCE_GPU_SUCCESS(cudaEventRecord(event_put_[bucket], stream_put_));
  return num_signals;
}

// Expert PUT (stream_put_): wait on the mark -> this rank's block for group
// rank `peer` goes to row `rank` of the peer's expert segment, incrementing its
// signal counter of the bucket -> record the event. Returns how much the peers'
// signal counters grow.
int GinGradCommScheduler::SubmitExpertPut(int bucket) {
  const Group& group = groups_[kGinExpertBucket];
  const GinBucketDesc& bucket_desc = buckets_[bucket];
  const int64_t block_bytes = bucket_desc.shard_numel * kFloatBytes;
  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaStreamWaitEvent(stream_put_, current_mark_, 0));
  int num_signals = 0;
  if (group.size > 1) {
    std::vector<PutArgs> puts;
    puts.reserve(static_cast<size_t>(group.size - 1));
    for (int peer_index = 0; peer_index < group.size - 1; ++peer_index) {
      const int peer = (group.rank + 1 + peer_index) % group.size;
      PutArgs put;
      put.src_window = group.windows.send_window;
      put.src_offset =
          static_cast<size_t>(bucket_desc.send_offset + peer * block_bytes);
      put.dst_window = group.windows.recv_window;
      put.dst_offset = static_cast<size_t>(bucket_desc.recv_offset[0] +
                                           group.rank * block_bytes);
      put.bytes = static_cast<size_t>(block_bytes);
      put.peer = peer;
      put.counter_window = group.windows.host_window;
      put.counter_offset = static_cast<size_t>(bucket_desc.signal_offset);
      puts.push_back(put);
    }
    group.pg->Put(puts, stream_put_);
    num_signals = group.size - 1;
  }
  PADDLE_ENFORCE_GPU_SUCCESS(cudaEventRecord(event_put_[bucket], stream_put_));
  return num_signals;
}

// Dense FINAL_SUM (stream_final_sum_): wait on the bucket's INTRA_SUM event
// and on the mark -> this rank's shard = sum over nodes of the partial sums *
// scale -> an inter-node credit goes to the ranks with the same rank_in_node
// on the other nodes -> record the event.
void GinGradCommScheduler::SubmitDenseFinalSum(int bucket) {
  const Group& group = groups_[kGinDenseBucket];
  const GinBucketDesc& bucket_desc = buckets_[bucket];
  const int64_t block_bytes = bucket_desc.shard_numel * kFloatBytes;
  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaStreamWaitEvent(stream_final_sum_, event_intra_sum_[bucket], 0));
  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaStreamWaitEvent(stream_final_sum_, current_mark_, 0));
  const int64_t shard_index =
      static_cast<int64_t>(group.node_index) * group.ranks_per_node +
      group.rank_in_node;
  PADDLE_ENFORCE_GPU_SUCCESS(LaunchFinalSum(
      reinterpret_cast<float*>(group.send + bucket_desc.send_offset +
                               shard_index * block_bytes),
      reinterpret_cast<const float*>(group.recv + bucket_desc.recv_offset[1]),
      static_cast<size_t>(bucket_desc.shard_numel),
      group.num_nodes,
      group.node_index,
      scale_[bucket],
      final_sum_blocks_,
      stream_final_sum_));
  if (group.num_nodes > 1) {
    std::vector<IncrementPeerCounterArgs> credits;
    credits.reserve(static_cast<size_t>(group.num_nodes - 1));
    for (int node = 0; node < group.num_nodes; ++node) {
      if (node == group.node_index) continue;
      IncrementPeerCounterArgs credit;
      credit.peer = node * group.ranks_per_node + group.rank_in_node;
      credit.counter_window = group.windows.host_window;
      credit.counter_offset = static_cast<size_t>(
          credit_inter_offset_ + kCounterBytes * group.node_index);
      credits.push_back(credit);
    }
    group.pg->IncrementPeerCounter(credits, stream_final_sum_);
  }
  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaEventRecord(event_final_sum_[bucket], stream_final_sum_));
}

// Expert FINAL_SUM (stream_final_sum_): wait on the mark -> this rank's shard
// = sum over the expert group * scale -> an expert credit goes to every other
// rank of the group -> record the event.
void GinGradCommScheduler::SubmitExpertFinalSum(int bucket) {
  const Group& group = groups_[kGinExpertBucket];
  const GinBucketDesc& bucket_desc = buckets_[bucket];
  const int64_t block_bytes = bucket_desc.shard_numel * kFloatBytes;
  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaStreamWaitEvent(stream_final_sum_, current_mark_, 0));
  PADDLE_ENFORCE_GPU_SUCCESS(LaunchFinalSum(
      reinterpret_cast<float*>(group.send + bucket_desc.send_offset +
                               group.rank * block_bytes),
      reinterpret_cast<const float*>(group.recv + bucket_desc.recv_offset[0]),
      static_cast<size_t>(bucket_desc.shard_numel),
      group.size,
      group.rank,
      scale_[bucket],
      final_sum_blocks_,
      stream_final_sum_));
  if (group.size > 1) {
    std::vector<IncrementPeerCounterArgs> credits;
    credits.reserve(static_cast<size_t>(group.size - 1));
    for (int group_rank = 0; group_rank < group.size; ++group_rank) {
      if (group_rank == group.rank) continue;
      IncrementPeerCounterArgs credit;
      credit.peer = group_rank;
      credit.counter_window = group.windows.host_window;
      credit.counter_offset = static_cast<size_t>(credit_expert_offset_ +
                                                  kCounterBytes * group.rank);
      credits.push_back(credit);
    }
    group.pg->IncrementPeerCounter(credits, stream_final_sum_);
  }
  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaEventRecord(event_final_sum_[bucket], stream_final_sum_));
}

// GATHER, sender side (stream_gather_put_): wait on `wait_event` -> this
// rank's shard goes to the same offset of the owner's send window (within the
// node: CopyToPeers, then this rank's same-node arrival counter of the bucket
// on the owner is incremented by 1; across nodes: one Put that increments the
// owner's inter-node arrival counter of the bucket by 1) -> record the event.
// Each sender sends each 2D bucket once per step, so every gather adds
// exactly 1 to one counter of the owner.
void GinGradCommScheduler::SubmitGatherPut(int entry, cudaEvent_t wait_event) {
  const GatherEntry& gather = gather_[entry];
  const int bucket = gather.gather_desc.bucket;
  const GinBucketDesc& bucket_desc = buckets_[bucket];
  const Group& group = GroupOf(bucket);
  const int64_t shard_bytes = bucket_desc.shard_numel * kFloatBytes;
  const size_t offset =
      static_cast<size_t>(bucket_desc.send_offset + group.rank * shard_bytes);
  const int owner = gather.gather_desc.owner;
  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaStreamWaitEvent(stream_gather_put_, wait_event, 0));
  if (owner / group.ranks_per_node == group.node_index) {
    CopyToPeersArgs copy;
    copy.src_window = group.windows.send_window;
    copy.src_offset = offset;
    copy.dst_window = group.windows.send_window;
    copy.dst_offset = offset;
    copy.peer = owner;
    copy.bytes = static_cast<size_t>(shard_bytes);
    group.pg->CopyToPeers({copy}, stream_gather_put_);
    IncrementPeerCounterArgs arrival;
    arrival.peer = owner;
    arrival.counter_window = group.windows.host_window;
    arrival.counter_offset =
        static_cast<size_t>(gather.gather_desc.intra_arrive_offset +
                            kCounterBytes * group.rank_in_node);
    group.pg->IncrementPeerCounter({arrival}, stream_gather_put_);
  } else {
    PutArgs put;
    put.src_window = group.windows.send_window;
    put.src_offset = offset;
    put.dst_window = group.windows.send_window;
    put.dst_offset = offset;
    put.bytes = static_cast<size_t>(shard_bytes);
    put.peer = owner;
    put.counter_window = group.windows.host_window;
    put.counter_offset = static_cast<size_t>(gather.gather_desc.arrive_offset);
    group.pg->Put({put}, stream_gather_put_);
  }
  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaEventRecord(event_gather_put_[entry], stream_gather_put_));
  gather_put_entries_.push_back(entry);
}

// `stream` waits on the final events of one bucket: INTRA_COPY, PUT and
// FINAL_SUM for a dense bucket, PUT and FINAL_SUM for an expert bucket.
void GinGradCommScheduler::JoinBucket(int bucket, cudaStream_t stream) const {
  if (buckets_[bucket].kind == kGinDenseBucket) {
    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaStreamWaitEvent(stream, event_intra_copy_[bucket], 0));
  }
  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaStreamWaitEvent(stream, event_put_[bucket], 0));
  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaStreamWaitEvent(stream, event_final_sum_[bucket], 0));
}

// Runs one submission with activity_ set; on failure records the op, the
// bucket and the reason, and raises with them.
void GinGradCommScheduler::Submit(int op,
                                  int bucket,
                                  const std::function<void()>& submit) {
  activity_ = op * kActivityScale + bucket + 1;
  try {
    submit();
  } catch (const std::exception& e) {
    activity_ = 0;
    Fail(op, bucket, e.what());
    PADDLE_THROW(common::errors::External("%s", ErrorText()));
  }
  activity_ = 0;
}

void GinGradCommScheduler::Fail(int op, int bucket, const std::string& what) {
  err_op_ = op;
  err_bucket_ = bucket;
  err_what_ = what;
}

std::string GinGradCommScheduler::ErrorText() const {
  std::ostringstream text;
  text << "gin_overlap: step " << step_ + 1 << ": submitting "
       << OpName(err_op_) << " of bucket " << err_bucket_
       << " failed: " << err_what_;
  return text.str();
}

// The release functions: starting at the next unreleased bucket of the fixed
// order, release every bucket that can go and stop at the first one that
// cannot. Release() and the proxy thread both call them; they differ only in
// current_mark_ and the block counts.

bool GinGradCommScheduler::ReleaseIntraCopy() {
  bool released = false;
  const int num_dense = static_cast<int>(dense_order_.size());
  while (pos_intra_copy_ < num_dense) {
    const int bucket = dense_order_[pos_intra_copy_];
    // gradients not complete yet, or the receivers' slot is not free
    if (!ready_[bucket] || BlockedIntraCopy(bucket, nullptr)) break;
    // the INTRA_SUM target of this bucket (see BlockedIntraSum)
    ++copy_seq_[bucket];
    Submit(kGinOpIntraCopy, bucket, [&] { SubmitIntraCopy(bucket); });
    ++pos_intra_copy_;
    released = true;
  }
  return released;
}

bool GinGradCommScheduler::ReleaseIntraSum() {
  bool released = false;
  const int num_dense = static_cast<int>(dense_order_.size());
  while (pos_intra_sum_ < num_dense) {
    const int bucket = dense_order_[pos_intra_sum_];
    // the bucket's INTRA_COPY is released and the rows of the other ranks of
    // the node have arrived
    if (pos_intra_sum_ >= pos_intra_copy_ || BlockedIntraSum(bucket, nullptr)) {
      break;
    }
    Submit(kGinOpIntraSum, bucket, [&] { SubmitIntraSum(bucket); });
    ++pos_intra_sum_;
    released = true;
  }
  return released;
}

bool GinGradCommScheduler::ReleasePut() {
  bool released = false;
  const int num_dense = static_cast<int>(dense_order_.size());
  while (pos_dense_put_ < num_dense) {
    const int bucket = dense_order_[pos_dense_put_];
    // the bucket's INTRA_SUM is released and its slot in the peers'
    // inter-node segment is free
    if (pos_dense_put_ >= pos_intra_sum_ || BlockedDensePut(bucket, nullptr)) {
      break;
    }
    int num_signals = 0;
    Submit(kGinOpPut, bucket, [&] { num_signals = SubmitDensePut(bucket); });
    targets_[bucket] += static_cast<uint64_t>(num_signals);
    ++pos_dense_put_;
    released = true;
  }
  const int num_expert = static_cast<int>(expert_order_.size());
  while (pos_expert_put_ < num_expert) {
    const int bucket = expert_order_[pos_expert_put_];
    // gradients not complete yet, or the peers' slot is not free
    if (!ready_[bucket] || BlockedExpertPut(bucket, nullptr)) break;
    int num_signals = 0;
    Submit(kGinOpPut, bucket, [&] { num_signals = SubmitExpertPut(bucket); });
    targets_[bucket] += static_cast<uint64_t>(num_signals);
    ++pos_expert_put_;
    released = true;
  }
  return released;
}

// in_proxy: the proxy thread sends the gather of a 2D bucket right after its
// FINAL_SUM; otherwise the bucket waits for the next GATHER.
bool GinGradCommScheduler::ReleaseFinalSum(bool in_proxy) {
  bool released = false;
  const int num_dense = static_cast<int>(dense_order_.size());
  while (pos_dense_final_sum_ < num_dense) {
    const int bucket = dense_order_[pos_dense_final_sum_];
    // the bucket's PUT is released and the puts of all other ranks arrived
    if (pos_dense_final_sum_ >= pos_dense_put_ ||
        BlockedDenseFinalSum(bucket, nullptr)) {
      break;
    }
    Submit(kGinOpFinalSum, bucket, [&] { SubmitDenseFinalSum(bucket); });
    ++pos_dense_final_sum_;
    AfterFinalSum(bucket, in_proxy);
    released = true;
  }
  const int num_expert = static_cast<int>(expert_order_.size());
  while (pos_expert_final_sum_ < num_expert) {
    const int bucket = expert_order_[pos_expert_final_sum_];
    if (pos_expert_final_sum_ >= pos_expert_put_ ||
        BlockedExpertFinalSum(bucket, nullptr)) {
      break;
    }
    Submit(kGinOpFinalSum, bucket, [&] { SubmitExpertFinalSum(bucket); });
    ++pos_expert_final_sum_;
    AfterFinalSum(bucket, in_proxy);
    released = true;
  }
  return released;
}

// After the FINAL_SUM of a 2D bucket. Nothing is sent by the owner. In the
// proxy thread the gather goes out at once and waits only on the FINAL_SUM
// event; otherwise the bucket waits in gather_pending_ for ReleaseGather().
void GinGradCommScheduler::AfterFinalSum(int bucket, bool in_proxy) {
  const int entry = gather_index_[bucket];
  if (entry < 0) return;
  if (gather_[entry].is_owner) return;
  if (in_proxy) {
    Submit(kGinOpGather, bucket, [&] {
      SubmitGatherPut(entry, event_final_sum_[bucket]);
    });
  } else {
    gather_pending_.push_back(entry);
  }
}

// GATHER (caller thread only): sends the gathers of the 2D buckets whose
// FINAL_SUM this thread released, in release order. The gate stream waits on
// `mark` (none when null), then per bucket on its final events, and records
// the bucket's source event, which the gather put waits on. No remote
// condition: the target range on the owner is fixed.
bool GinGradCommScheduler::ReleaseGather(cudaEvent_t mark) {
  if (gather_pending_.empty()) return false;
  if (mark != nullptr) {
    Submit(
        kGinOpGather, gather_[gather_pending_.front()].gather_desc.bucket, [&] {
          PADDLE_ENFORCE_GPU_SUCCESS(
              cudaStreamWaitEvent(stream_gather_gate_, mark, 0));
        });
  }
  for (const int entry : gather_pending_) {
    const int bucket = gather_[entry].gather_desc.bucket;
    Submit(kGinOpGather, bucket, [&] {
      JoinBucket(bucket, stream_gather_gate_);
      PADDLE_ENFORCE_GPU_SUCCESS(
          cudaEventRecord(event_gather_source_[entry], stream_gather_gate_));
      SubmitGatherPut(entry, event_gather_source_[entry]);
    });
  }
  gather_pending_.clear();
  return true;
}

bool GinGradCommScheduler::IsStepDone() const {
  const int num_dense = static_cast<int>(dense_order_.size());
  const int num_expert = static_cast<int>(expert_order_.size());
  return pos_intra_copy_ >= num_dense && pos_intra_sum_ >= num_dense &&
         pos_dense_put_ >= num_dense && pos_dense_final_sum_ >= num_dense &&
         pos_expert_put_ >= num_expert && pos_expert_final_sum_ >= num_expert;
}

// Buckets that reached an op but are not released, in the order [intra copy,
// intra sum, put, final sum] (put and final sum count dense and expert).
// Intra copy and expert put count only ready buckets.
void GinGradCommScheduler::PendingCounts(int* counts) const {
  int ready_copy = 0;
  int ready_expert_put = 0;
  for (size_t position = pos_intra_copy_; position < dense_order_.size();
       ++position) {
    if (ready_[dense_order_[position]]) ++ready_copy;
  }
  for (size_t position = pos_expert_put_; position < expert_order_.size();
       ++position) {
    if (ready_[expert_order_[position]]) ++ready_expert_put;
  }
  counts[0] = ready_copy;
  counts[1] = pos_intra_copy_ - pos_intra_sum_;
  counts[2] = pos_intra_sum_ - pos_dense_put_ + ready_expert_put;
  counts[3] = pos_dense_put_ - pos_dense_final_sum_ + pos_expert_put_ -
              pos_expert_final_sum_;
}

// Records the next tail mark of this step on the compute stream and returns
// its slot. The slots of a step are used in order from 0, each once.
int GinGradCommScheduler::RecordTailMark() {
  PADDLE_ENFORCE_LT(
      tail_mark_index_,
      static_cast<int>(tail_marks_.size()),
      common::errors::ResourceExhausted(
          "Step %d used all %d tail marks (one per hand-over, per bucket "
          "reported after it and per ReleaseAsync() after it); raise "
          "num_tail_marks.",
          step_ + 1,
          static_cast<int>(tail_marks_.size())));
  const int slot = tail_mark_index_;
  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaEventRecord(tail_marks_[slot], ComputeStream(device_id_)));
  ++tail_mark_index_;
  return slot;
}

// Hands release over to the proxy thread (mutex_ held, state idle).
void GinGradCommScheduler::StartProxy(int slot,
                                      int intra_sum_blocks,
                                      int final_sum_blocks) {
  current_mark_ = tail_marks_[slot];
  slot_bucket_[slot] = kSlotMarkOnly;
  consumed_slot_ = slot;
  intra_sum_blocks_ = intra_sum_blocks;
  final_sum_blocks_ = final_sum_blocks;
  abort_ = false;
  state_ = kRun;
  cv_proxy_.notify_all();
}

void GinGradCommScheduler::BucketReady(int bucket, float scale) {
  PADDLE_ENFORCE_EQ(
      bucket >= 0 && bucket < static_cast<int>(buckets_.size()),
      true,
      common::errors::InvalidArgument("Bucket %d is out of range [0, %d).",
                                      bucket,
                                      static_cast<int>(buckets_.size())));
  std::lock_guard<std::mutex> lock(mutex_);
  if (state_ == kErr) {
    PADDLE_THROW(common::errors::External("%s", ErrorText()));
  }
  PADDLE_ENFORCE_EQ(
      reported_[bucket] == 0,
      true,
      common::errors::PreconditionNotMet(
          "Bucket %d was reported twice in step %d.", bucket, step_ + 1));
  if (state_ == kIdle) {
    // The caller releases: record the bucket for a later Release().
    reported_[bucket] = 1;
    ++num_reported_;
    ready_[bucket] = 1;
    scale_[bucket] = scale;
    return;
  }
  PADDLE_ENFORCE_EQ(state_ == kRun,
                    true,
                    common::errors::PreconditionNotMet(
                        "Bucket %d was reported in step %d after the whole "
                        "step was released (state %d).",
                        bucket,
                        step_ + 1,
                        state_));
  // The proxy thread releases: publish the bucket under a tail mark recorded
  // now; the slot entry, which makes the bucket ready, is written last.
  const int slot = RecordTailMark();
  reported_[bucket] = 1;
  ++num_reported_;
  published_scale_[bucket] = scale;
  slot_bucket_[slot] = bucket;
  cv_proxy_.notify_all();
}

void GinGradCommScheduler::Release(int ops,
                                   int intra_sum_blocks,
                                   int final_sum_blocks) {
  PADDLE_ENFORCE_EQ(ops != 0 && (ops & ~kAllOps) == 0,
                    true,
                    common::errors::InvalidArgument(
                        "Release() got ops %d; ops is an OR of INTRA_COPY, "
                        "INTRA_SUM, PUT, FINAL_SUM and GATHER.",
                        ops));
  PADDLE_ENFORCE_EQ(
      intra_sum_blocks >= 0 && final_sum_blocks >= 0 &&
          (intra_sum_blocks > 0 || (ops & kGinOpIntraSum) == 0) &&
          (final_sum_blocks > 0 || (ops & kGinOpFinalSum) == 0),
      true,
      common::errors::InvalidArgument(
          "Release() got intra_sum_blocks %d and final_sum_blocks %d; each "
          "must be positive when its op (INTRA_SUM / FINAL_SUM) is selected, "
          "and not negative otherwise.",
          intra_sum_blocks,
          final_sum_blocks));
  std::lock_guard<std::mutex> lock(mutex_);
  if (state_ != kIdle) return;
  const int slot = window_mark_index_;
  window_mark_index_ = (slot + 1) % static_cast<int>(window_marks_.size());
  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaEventRecord(window_marks_[slot], ComputeStream(device_id_)));
  current_mark_ = window_marks_[slot];
  intra_sum_blocks_ = intra_sum_blocks;
  final_sum_blocks_ = final_sum_blocks;
  // pipeline order
  if (ops & kGinOpIntraCopy) ReleaseIntraCopy();
  if (ops & kGinOpIntraSum) ReleaseIntraSum();
  if (ops & kGinOpPut) ReleasePut();
  if (ops & kGinOpFinalSum) ReleaseFinalSum(/*in_proxy=*/false);
  if (ops & kGinOpGather) ReleaseGather(current_mark_);
}

void GinGradCommScheduler::ReleaseAsync(int intra_sum_blocks,
                                        int final_sum_blocks) {
  PADDLE_ENFORCE_EQ(intra_sum_blocks > 0 && final_sum_blocks > 0,
                    true,
                    common::errors::InvalidArgument(
                        "ReleaseAsync() got intra_sum_blocks %d and "
                        "final_sum_blocks %d; both must be positive.",
                        intra_sum_blocks,
                        final_sum_blocks));
  std::lock_guard<std::mutex> lock(mutex_);
  if (state_ == kIdle) {
    StartProxy(RecordTailMark(), intra_sum_blocks, final_sum_blocks);
  } else if (state_ == kRun) {
    // a later point of the compute stream for the proxy thread's next
    // submissions
    slot_bucket_[RecordTailMark()] = kSlotMarkOnly;
    cv_proxy_.notify_all();
  }
}

// Proxy thread, mutex_ held: takes the next tail mark slot after the last
// one taken, in the order the slots were recorded. A slot of BucketReady()
// makes its bucket ready with its scale; every slot becomes the mark of the
// following submissions. Returns whether it took one.
bool GinGradCommScheduler::ConsumePublished() {
  const int next_slot = consumed_slot_ + 1;
  if (next_slot >= tail_mark_index_) return false;
  const int bucket = slot_bucket_[next_slot];
  if (bucket >= 0) {
    ready_[bucket] = 1;
    scale_[bucket] = published_scale_[bucket];
  }
  current_mark_ = tail_marks_[next_slot];
  consumed_slot_ = next_slot;
  return true;
}

// One step of the proxy thread (in kRun only this thread changes the release
// state): take one report -> run passes until none makes progress -> return
// kDone once everything is released. With neither a report nor progress, wait
// on the condition variable for at most kPollIntervalUs. A pass tries
// FINAL_SUM first, since it frees recv pool slots. CUDA calls are made outside
// the lock.
int GinGradCommScheduler::RunStep() {
  for (;;) {
    bool took_mark = false;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (abort_) return kAborted;
      took_mark = ConsumePublished();
    }
    bool released = false;
    for (;;) {
      bool released_in_pass = ReleaseFinalSum(/*in_proxy=*/true);
      released_in_pass |= ReleaseIntraCopy();
      released_in_pass |= ReleaseIntraSum();
      released_in_pass |= ReleasePut();
      if (!released_in_pass) break;
      released = true;
    }
    if (IsStepDone()) return kDone;
    if (took_mark || released) continue;
    std::unique_lock<std::mutex> lock(mutex_);
    if (consumed_slot_ + 1 >= tail_mark_index_ && !abort_) {
      cv_proxy_.wait_for(lock, std::chrono::microseconds(kPollIntervalUs));
    }
  }
}

void GinGradCommScheduler::ProxyMain() {
  // same device (primary context) as the caller
  const cudaError_t status = cudaSetDevice(device_id_);
  // timer slack of 1 us, so that the short waits wake up on time
  prctl(PR_SET_TIMERSLACK, 1000UL, 0UL, 0UL, 0UL);
  std::unique_lock<std::mutex> lock(mutex_);
  proxy_device_status_ = static_cast<int>(status);
  proxy_started_ = true;
  cv_main_.notify_all();
  if (status != cudaSuccess) return;
  for (;;) {
    cv_proxy_.wait(lock, [this] { return state_ == kRun || stop_; });
    if (stop_) return;
    lock.unlock();
    int state = kErr;
    try {
      state = RunStep();
    } catch (const std::exception& e) {
      // Submit() recorded the failure; anything else is recorded here.
      if (err_what_.empty()) Fail(0, -1, e.what());
    }
    lock.lock();
    state_ = state;
    cv_main_.notify_all();
  }
}

void GinGradCommScheduler::FinishStep(double timeout_s) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (state_ == kErr) {
    const std::string text = ErrorText();
    state_ = kIdle;
    PADDLE_THROW(common::errors::External("%s", text));
  }
  const int num_buckets = static_cast<int>(buckets_.size());
  if (num_reported_ != num_buckets) {
    std::ostringstream missing;
    int listed = 0;
    for (int bucket = 0; bucket < num_buckets && listed < kStuckMaxItems;
         ++bucket) {
      if (reported_[bucket]) continue;
      missing << (listed++ ? ", " : "") << bucket;
    }
    PADDLE_THROW(common::errors::PreconditionNotMet(
        "gin_overlap: %d of %d buckets were not reported in step %d (e.g. "
        "%s). Every bucket must be reported every step; a bucket whose "
        "parameters are all frozen is never reported. The cause may also be "
        "that FusedCommBuffer._comm_grads was replaced (for example by "
        "PaddleFleet's accuracy_compatible patch, use_dsv4_accuracy), so the "
        "buckets never reach the scheduler.",
        num_buckets - num_reported_,
        num_buckets,
        step_ + 1,
        missing.str()));
  }
  // Hand over here when the step never called ReleaseAsync() and the caller
  // has not released everything.
  if (state_ == kIdle && !IsStepDone()) {
    StartProxy(
        RecordTailMark(), kFallbackIntraSumBlocks, kFallbackFinalSumBlocks);
  }
  if (state_ == kRun) {
    const auto left_run = [this] { return state_ != kRun; };
    cv_main_.wait_for(lock, std::chrono::duration<double>(timeout_s), left_run);
    if (state_ == kRun) {
      // The proxy thread checks abort_ between passes; stuck in a CUDA call
      // it cannot stop.
      abort_ = true;
      cv_proxy_.notify_all();
      cv_main_.wait_for(
          lock, std::chrono::duration<double>(kAbortWaitSeconds), left_run);
      if (state_ == kRun) {
        const int activity = activity_.load();
        std::ostringstream activity_text;
        if (activity) {
          activity_text << "submitting " << OpName(activity / kActivityScale)
                        << " of bucket " << activity % kActivityScale - 1;
        } else {
          activity_text << "not in a CUDA call";
        }
        std::ostringstream text;
        text << "gin_overlap: step " << step_ + 1 << " was not released within "
             << timeout_s << " s, and the proxy thread did not stop within "
             << kAbortWaitSeconds << " s (" << activity_text.str()
             << "), so there is no per-op report.";
        LOG(ERROR) << text.str();
        PADDLE_THROW(common::errors::ExecutionTimeout("%s", text.str()));
      }
      state_ = kIdle;
      abort_ = false;
      std::ostringstream text;
      text << "gin_overlap: step " << step_ + 1 << " was not released within "
           << timeout_s << " s: " << StuckReportLocked();
      LOG(ERROR) << text.str();
      PADDLE_THROW(common::errors::ExecutionTimeout("%s", text.str()));
    }
  }
  if (state_ == kErr) {
    const std::string text = ErrorText();
    state_ = kIdle;
    PADDLE_THROW(common::errors::External("%s", text));
  }
  state_ = kIdle;
  abort_ = false;
  lock.unlock();

  // The release state is the caller's again. Gathers of 2D buckets whose
  // FINAL_SUM the caller released but no later GATHER sent go out now,
  // without a mark.
  ReleaseGather(nullptr);
  const cudaStream_t compute_stream = ComputeStream(device_id_);
  for (int bucket = 0; bucket < num_buckets; ++bucket) {
    JoinBucket(bucket, compute_stream);
  }
  WaitOwnedGathers(timeout_s);
  for (const int entry : gather_put_entries_) {
    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaStreamWaitEvent(compute_stream, event_gather_put_[entry], 0));
  }
  // The send window must not be reused by the next step before this flush.
  groups_[kGinDenseBucket].pg->Flush(compute_stream);
  if (!expert_order_.empty()) {
    groups_[kGinExpertBucket].pg->Flush(compute_stream);
  }
  lock.lock();
  ResetStep();
  ++step_;
}

// Owner side of GATHER: waits until every 2D bucket this rank owns has all of
// its shards, reading the arrival counters of the host window. Every gather
// adds 1 to one counter of the owner, and each sender sends each bucket once
// per step. The inter-node counter of a bucket is shared by all senders on
// other nodes, so after s steps it is s * (senders on other nodes); the
// same-node counter of node-local rank `local_rank` has one sender, that rank,
// so it is s. A larger value means a gather was sent twice and raises.
void GinGradCommScheduler::WaitOwnedGathers(double timeout_s) {
  std::vector<int> pending;
  for (size_t entry = 0; entry < gather_.size(); ++entry) {
    if (!gather_[entry].is_owner) continue;
    ++gather_owner_step_[entry];
    pending.push_back(static_cast<int>(entry));
  }
  const auto deadline =
      std::chrono::steady_clock::now() +
      std::chrono::duration_cast<std::chrono::steady_clock::duration>(
          std::chrono::duration<double>(timeout_s));
  std::string why;
  while (!pending.empty()) {
    std::vector<int> still_pending;
    for (const int entry : pending) {
      if (OwnedGatherPending(entry, nullptr)) still_pending.push_back(entry);
    }
    const bool some_complete = still_pending.size() != pending.size();
    pending.swap(still_pending);
    if (pending.empty() || some_complete) continue;
    if (std::chrono::steady_clock::now() > deadline) {
      std::ostringstream text;
      text << "gin_overlap: step " << step_ + 1 << ": the gather of "
           << pending.size() << " 2D buckets owned by this rank did not "
           << "complete within " << timeout_s << " s:";
      for (size_t pending_index = 0;
           pending_index < pending.size() &&
           pending_index < static_cast<size_t>(kStuckMaxItems);
           ++pending_index) {
        OwnedGatherPending(pending[pending_index], &why);
        text << " " << why << ";";
      }
      LOG(ERROR) << text.str();
      PADDLE_THROW(common::errors::ExecutionTimeout("%s", text.str()));
    }
    std::this_thread::sleep_for(std::chrono::microseconds(kGatherPollUs));
  }
}

// Whether the owner still waits for shards of the 2D bucket of `entry`; `why`
// (may be null) receives the counter it waits for.
bool GinGradCommScheduler::OwnedGatherPending(int entry,
                                              std::string* why) const {
  const GatherEntry& gather = gather_[entry];
  const int bucket = gather.gather_desc.bucket;
  const Group& group = GroupOf(bucket);
  const uint64_t steps = static_cast<uint64_t>(gather_owner_step_[entry]);
  const uint64_t inter_target =
      steps * static_cast<uint64_t>(gather.num_inter_senders);
  // Whether `value` is still below `target`. A value above its target means
  // a gather was sent twice. `sender` is the group rank of a same-node
  // sender, or -1 for the inter-node counter (all senders on other nodes).
  const auto below_target = [&](uint64_t value, uint64_t target, int sender) {
    if (value <= target && (value == target || !why)) return value < target;
    std::ostringstream counter_text;
    if (sender < 0) {
      counter_text
          << "inter-node arrival counter (sum of all senders on other nodes)";
    } else {
      counter_text << "same-node arrival counter of group rank " << sender;
    }
    PADDLE_ENFORCE_LE(
        value,
        target,
        common::errors::PreconditionNotMet(
            "gin_overlap: owner of 2D bucket %d in step %d: the %s is %d, "
            "above its target %d; a gather was sent twice.",
            bucket,
            steps,
            counter_text.str(),
            value,
            target));
    std::ostringstream text;
    text << "bucket " << bucket << " waits for the " << counter_text.str()
         << ": current " << value << " / target " << target;
    *why = text.str();
    return true;
  };
  if (below_target(HostCounter(group, gather.gather_desc.arrive_offset),
                   inter_target,
                   -1)) {
    return true;
  }
  for (const int local_rank : gather.intra_senders) {
    if (below_target(HostCounter(group,
                                 gather.gather_desc.intra_arrive_offset +
                                     kCounterBytes * local_rank),
                     steps,
                     group.node_index * group.ranks_per_node + local_rank)) {
      return true;
    }
  }
  return false;
}

void GinGradCommScheduler::ResetStep() {
  pos_intra_copy_ = 0;
  pos_intra_sum_ = 0;
  pos_dense_put_ = 0;
  pos_dense_final_sum_ = 0;
  pos_expert_put_ = 0;
  pos_expert_final_sum_ = 0;
  std::fill(ready_.begin(), ready_.end(), 0);
  std::fill(reported_.begin(), reported_.end(), 0);
  std::fill(slot_bucket_.begin(), slot_bucket_.end(), kSlotUnused);
  gather_pending_.clear();
  gather_put_entries_.clear();
  num_reported_ = 0;
  consumed_slot_ = -1;
  tail_mark_index_ = 0;
  current_mark_ = nullptr;
  err_op_ = 0;
  err_bucket_ = -1;
  err_what_.clear();
}

int64_t GinGradCommScheduler::CompletedSteps() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return step_;
}

bool GinGradCommScheduler::HasExpertBuckets() const {
  return !expert_order_.empty();
}

GinGradCommStatus GinGradCommScheduler::Status() const {
  std::lock_guard<std::mutex> lock(mutex_);
  GinGradCommStatus status;
  status.state = state_;
  status.completed_steps = step_;
  status.reported = num_reported_;
  // The release state belongs to the proxy thread while it runs.
  if (state_ == kRun) return status;
  // Same order as GinGradCommStatus::positions documents.
  status.positions = {pos_intra_copy_,
                      pos_intra_sum_,
                      pos_dense_put_,
                      pos_dense_final_sum_,
                      pos_expert_put_,
                      pos_expert_final_sum_};
  int pending[4] = {0, 0, 0, 0};
  PendingCounts(pending);
  status.pending.assign(pending, pending + 4);
  return status;
}

std::string GinGradCommScheduler::StuckReport() const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (state_ == kRun) return "";
  return StuckReportLocked();
}

// Per op: "released/buckets", the next bucket of the fixed order when it has
// not reached the op, and what each bucket that reached the op but is not
// released waits for (at most kStuckMaxItems per op). Reads the host windows
// only.
std::string GinGradCommScheduler::StuckReportLocked() const {
  struct OpRow {
    const char* name;
    bool expert;
    int position;
    int previous;  // position of the previous op, -1 = use ready_
    const char* not_reached;
    int which;
  };
  const OpRow rows[] = {
      {"dense intra copy", false, pos_intra_copy_, -1, "is not ready yet", 0},
      {"dense intra sum",
       false,
       pos_intra_sum_,
       pos_intra_copy_,
       "has not had its intra copy released",
       1},
      {"dense put",
       false,
       pos_dense_put_,
       pos_intra_sum_,
       "has not had its intra sum released",
       2},
      {"dense final sum",
       false,
       pos_dense_final_sum_,
       pos_dense_put_,
       "has not had its put released",
       3},
      {"expert put", true, pos_expert_put_, -1, "is not ready yet", 4},
      {"expert final sum",
       true,
       pos_expert_final_sum_,
       pos_expert_put_,
       "has not had its put released",
       5},
  };
  std::ostringstream text;
  bool first_row = true;
  for (const OpRow& row : rows) {
    const std::vector<int>& order = row.expert ? expert_order_ : dense_order_;
    const int num_buckets = static_cast<int>(order.size());
    text << (first_row ? "" : "; ") << row.name << " " << row.position << "/"
         << num_buckets;
    first_row = false;
    if (row.position >= num_buckets) {
      text << " all released";
      continue;
    }
    std::vector<int> reached_buckets;
    for (int position = row.position; position < num_buckets; ++position) {
      const bool arrived = row.previous < 0 ? ready_[order[position]] != 0
                                            : position < row.previous;
      if (arrived) reached_buckets.push_back(order[position]);
    }
    if (reached_buckets.empty() || reached_buckets[0] != order[row.position]) {
      text << ", next bucket " << order[row.position] << " " << row.not_reached;
    }
    const int listed =
        std::min(static_cast<int>(reached_buckets.size()), kStuckMaxItems);
    for (int item_index = 0; item_index < listed; ++item_index) {
      const int bucket = reached_buckets[item_index];
      Blocker blocker;
      switch (row.which) {
        case 0:
          BlockedIntraCopy(bucket, &blocker);
          break;
        case 1:
          BlockedIntraSum(bucket, &blocker);
          break;
        case 2:
          BlockedDensePut(bucket, &blocker);
          break;
        case 3:
          BlockedDenseFinalSum(bucket, &blocker);
          break;
        case 4:
          BlockedExpertPut(bucket, &blocker);
          break;
        default:
          BlockedExpertFinalSum(bucket, &blocker);
          break;
      }
      text << (item_index == 0 ? ": " : ", ") << "bucket " << bucket;
      if (blocker.counter == 0) {
        // remote condition met, it only queues behind the previous bucket
        text << " is ready, queued behind the previous bucket of the order";
        continue;
      }
      text << " waits for " << CounterName(blocker.counter);
      if (blocker.sender >= 0) {
        text << " (node-local sender " << blocker.sender << ")";
      }
      if (blocker.peer >= 0) {
        text << " of peer group rank " << blocker.peer;
      } else {
        text << " (sum of all senders)";
      }
      text << ": current " << blocker.value << " / target " << blocker.target;
    }
    if (static_cast<int>(reached_buckets.size()) > listed) {
      text << ", ... " << static_cast<int>(reached_buckets.size()) - listed
           << " more";
    }
  }
  return text.str();
}

}  // namespace distributed
}  // namespace paddle
