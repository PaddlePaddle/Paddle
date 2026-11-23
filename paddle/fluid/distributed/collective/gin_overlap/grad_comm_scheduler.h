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

// GinGradCommScheduler: the reduce_scatter of the sharding-stage1 gradient
// buckets, split into stages built on the GIN primitives of ProcessGroupNCCL
// and released in the overlap windows of the backward pass.
//
// Terms
//   - bucket: a FusedCommBuffer; its grad_storage is a range of the send
//     window of its group. Buckets are numbered by one index across dense and
//     expert buckets (the bucket table).
//   - shard: the part of a bucket this rank owns after the reduce.
//   - window: memory registered on the communicator of a group. Each group
//     has a send window (the grad_storage of its buckets), a recv window (the
//     recv pool) and a host window (pinned host memory holding the u64
//     counters, read by the CPU).
//   - mark: a cudaEvent recorded on the compute stream. Work submitted to a
//     side stream waits on a mark first, so it never starts before the
//     gradients it reads are written.
//   - release: deciding that one op of one bucket may run now and submitting
//     it to the GPU.
//   - credit: after a receiver is done with a slot of its recv pool it adds 1
//     to a counter of the sender; the sender writes that slot again only once
//     the counter reaches its target.
//   - owner: the rank a Muon 2D bucket finally belongs to.
//   - proxy thread: the thread that releases everything left after
//     ReleaseAsync() (or the fallback of FinishStep()).
//
// Groups
//   - Dense buckets use the sharding group, expert buckets the moe_sharding
//     group. Both must be laid out node-major: the LSA team of the device
//     communicator is the node, rank_in_node = rank % ranks_per_node and
//     node_index = rank / ranks_per_node.
//
// Ops of a dense bucket (rank (node_index, rank_in_node) owns column
// rank_in_node, i.e. blocks {n * ranks_per_node + rank_in_node}):
//   1. INTRA_COPY: CopyToPeers writes this rank's blocks of column l into this
//      rank's row of the intra-node segment of node-local rank l, then
//      IncrementPeerCounter adds 1 to this rank's intra-node arrival counter
//      of the bucket on l.
//   2. INTRA_SUM: each block of this rank's column becomes the sum over the
//      node (LaunchIntraNodeSum, in place), then IncrementPeerCounter returns
//      an intra-node credit to the other ranks of the node.
//   3. PUT: block n of this column goes to the rank with the same
//      rank_in_node on node n, into row node_index of its inter-node segment,
//      incrementing its signal counter of the bucket.
//   4. FINAL_SUM: shard = sum over nodes * scale (LaunchFinalSum), then
//      IncrementPeerCounter returns an inter-node credit.
// An expert bucket has PUT (to every other rank of the expert group) and
// FINAL_SUM (with an expert credit). A Muon 2D bucket also has GATHER: after
// FINAL_SUM this rank's shard is written to the same offset of the owner's
// send window (CopyToPeers + IncrementPeerCounter within the node, Put across
// nodes; each adds 1 to an arrival counter of the owner); the owner waits for
// all shards in FinishStep().
//
// Rules
//   - Each op starts at the next unreleased bucket of the fixed release order
//     of its kind and releases every bucket it can; it stops at the first one
//     that cannot go. The orders must be the same on all ranks.
//   - Counters accumulate across steps and only grow by increments of 1; the
//     CPU never writes them. A receiver's target is the number of increments
//     the counter has received once the awaited writes have arrived.
//   - GPU work waits only on local events (marks, side-stream events); remote
//     counters are read on the CPU before release.
//   - Every bucket goes through all of its ops every step.
//   - The release state has one writer: the caller of Release() while the
//     scheduler is idle, the proxy thread while it runs.
#pragma once

#include <cuda_runtime.h>

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "paddle/common/macros.h"

namespace paddle {
namespace distributed {

class ProcessGroupNCCL;

// Ops of Release(), OR-ed together. The values are exported to Python.
enum GinGradCommOp : int {
  kGinOpIntraCopy = 1,
  kGinOpIntraSum = 2,
  kGinOpPut = 4,
  kGinOpFinalSum = 8,
  kGinOpGather = 16,
};

// Kind of a bucket of the bucket table.
enum GinBucketKind : int {
  kGinDenseBucket = 0,
  kGinExpertBucket = 1,
};

// Window handles of one group, as returned by register_comm_buffer() /
// alloc_host_window() of its ProcessGroupNCCL.
struct GinGroupWindows {
  void* send_window{nullptr};
  void* recv_window{nullptr};
  void* host_window{nullptr};
};

// One bucket of the bucket table. Offsets are bytes.
struct GinBucketDesc {
  int kind{kGinDenseBucket};
  // Range of the bucket in the send window of its group.
  int64_t send_offset{0};
  // u64 signal counter of the bucket in the host window, incremented by the
  // puts of the other ranks.
  int64_t signal_offset{0};
  // Elements of this rank's shard (bucket numel / group size).
  int64_t shard_numel{0};
  // Start of the 16 intra-node arrival counters of a dense bucket in the host
  // window; counter l gets 1 added by node-local rank l per INTRA_COPY of the
  // bucket (once per step). Unused for experts.
  int64_t arrive_offset{0};
  // Slot of the bucket in the recv window: dense = [intra-node segment,
  // inter-node segment], expert = [expert segment, unused].
  int64_t recv_offset[2]{0, 0};
  // Predecessor distance of each slot: how much smaller the release sequence
  // number of the previous holder of the slot is (1..n, n = the bucket itself
  // one step earlier); -1 = no slot, no credit to wait for.
  int64_t depth[2]{-1, -1};
};

// One Muon 2D bucket.
struct GinGatherDesc {
  int bucket{0};
  // Owner, a group rank of the bucket's group.
  int owner{0};
  // Owner's inter-node arrival counter in the host window (each gather put
  // from another node adds 1).
  int64_t arrive_offset{0};
  // Start of the owner's 16 same-node arrival counters of the bucket in the
  // host window (node-local sender l adds 1 to counter l per gather).
  int64_t intra_arrive_offset{0};
};

struct GinGradCommSchedulerConfig {
  // Sharding group of the dense buckets and moe_sharding group of the expert
  // buckets (nullptr without expert buckets). Their device communicators must
  // exist.
  ProcessGroupNCCL* dense_group{nullptr};
  ProcessGroupNCCL* expert_group{nullptr};
  GinGroupWindows dense_windows;
  GinGroupWindows expert_windows;
  // Indexed by bucket id.
  std::vector<GinBucketDesc> buckets;
  // Fixed release orders: permutations of the dense / expert bucket ids.
  std::vector<int> dense_order;
  std::vector<int> expert_order;
  // Counter areas in the host windows: 16 intra-node credits (dense), one
  // inter-node credit per node (dense), one expert credit per expert group
  // rank (expert).
  int64_t credit_intra_offset{0};
  int64_t credit_inter_offset{0};
  int64_t credit_expert_offset{0};
  // Muon 2D buckets; empty without Muon.
  std::vector<GinGatherDesc> gather;
  // Size of the mark ring of Release() and of the per-step tail mark ring of
  // the proxy thread.
  int num_window_marks{1024};
  int num_tail_marks{1024};
};

// Release progress, for debugging.
struct GinGradCommStatus {
  // 0 idle, 1 proxy running, 2 done, 3 error, 4 aborted.
  int state{0};
  int64_t completed_steps{0};
  // Positions in the fixed order, in this order: dense intra copy, dense
  // intra sum, dense put, dense final sum, expert put, expert final sum
  // (pos_intra_copy_ ... pos_expert_final_sum_ of the scheduler).
  std::vector<int> positions;
  // Buckets that reached an op but are not released: intra copy, intra sum,
  // put, final sum.
  std::vector<int> pending;
  // Buckets reported by BucketReady() in this step.
  int reported{0};
};

class GinGradCommScheduler {
 public:
  explicit GinGradCommScheduler(const GinGradCommSchedulerConfig& config);
  ~GinGradCommScheduler();

  // Bucket `bucket` is complete. `scale` is the factor of the reduce op (1/N
  // for AVG, 1 for SUM), applied in its FINAL_SUM. Raises on a second report
  // of the same bucket in this step. While idle it only records the bucket;
  // while the proxy thread runs it also records a mark on the compute stream
  // that the bucket's ops wait on, and wakes the proxy thread.
  void BucketReady(int bucket, float scale);

  // Records a mark on the compute stream and tries the selected ops once each
  // in pipeline order (INTRA_COPY, INTRA_SUM, PUT, FINAL_SUM, GATHER), without
  // waiting on remote counters. intra_sum_blocks / final_sum_blocks are the
  // grid sizes of the INTRA_SUM / FINAL_SUM kernels released by this call; a
  // count must be positive when its op is selected. Does nothing unless idle.
  // Raises with the op and bucket when a submission fails.
  void Release(int ops, int intra_sum_blocks, int final_sum_blocks);

  // Hands release over to the proxy thread, which loops over the ops until
  // the step is released; its INTRA_SUM / FINAL_SUM kernels use
  // intra_sum_blocks / final_sum_blocks blocks. Called again while the proxy
  // thread runs, it gives the proxy thread a new mark and leaves the block
  // counts unchanged.
  void ReleaseAsync(int intra_sum_blocks, int final_sum_blocks);

  // Ends the step: checks that every bucket was reported, hands over to the
  // proxy thread if that has not happened (its sums then use 16 / 64 blocks,
  // the counts of the pre window), and waits for the step to be
  // released (raises with a per-op report after timeout_s seconds). Then the
  // owner waits for its 2D buckets, the compute stream waits on the final
  // events of every bucket and on this step's gathers, the GIN contexts of
  // both groups are flushed on the compute stream, and the next step begins.
  void FinishStep(double timeout_s);

  // Number of steps finished by FinishStep().
  int64_t CompletedSteps() const;

  // Whether the bucket table has expert buckets.
  bool HasExpertBuckets() const;

  GinGradCommStatus Status() const;

  // Per op: buckets released, the next bucket, and what each bucket that
  // reached the op waits for. Empty reason while the proxy thread runs.
  std::string StuckReport() const;

 private:
  DISABLE_COPY_AND_ASSIGN(GinGradCommScheduler);

  struct Group {
    ProcessGroupNCCL* pg{nullptr};
    GinGroupWindows windows;
    char* send{nullptr};
    size_t send_bytes{0};
    char* recv{nullptr};
    size_t recv_bytes{0};
    const char* host{nullptr};
    size_t host_bytes{0};
    int size{0};
    int rank{0};
    int ranks_per_node{1};
    int rank_in_node{0};
    int num_nodes{1};
    int node_index{0};
  };

  // What a blocked op waits for.
  struct Blocker {
    int counter{0};
    int sender{-1};
    int peer{-1};
    uint64_t value{0};
    uint64_t target{0};
  };

  struct GatherEntry {
    GinGatherDesc gather_desc;
    bool is_owner{false};
    // Owner only: senders on other nodes and node-local indices of the
    // senders on this node.
    int num_inter_senders{0};
    std::vector<int> intra_senders;
  };

  void InitGroup(int kind,
                 ProcessGroupNCCL* pg,
                 const GinGroupWindows& windows);
  void CheckTables(const GinGradCommSchedulerConfig& config) const;
  void CreateStreamsAndEvents(int num_window_marks, int num_tail_marks);

  const Group& GroupOf(int bucket) const;
  uint64_t HostCounter(const Group& group, int64_t byte_offset) const;
  uint64_t CreditTarget(int num_buckets, int position, int64_t depth) const;

  bool BlockedIntraCopy(int bucket, Blocker* blocker) const;
  bool BlockedIntraSum(int bucket, Blocker* blocker) const;
  bool BlockedDensePut(int bucket, Blocker* blocker) const;
  bool BlockedExpertPut(int bucket, Blocker* blocker) const;
  bool BlockedDenseFinalSum(int bucket, Blocker* blocker) const;
  bool BlockedExpertFinalSum(int bucket, Blocker* blocker) const;

  void SubmitIntraCopy(int bucket);
  void SubmitIntraSum(int bucket);
  int SubmitDensePut(int bucket);
  int SubmitExpertPut(int bucket);
  void SubmitDenseFinalSum(int bucket);
  void SubmitExpertFinalSum(int bucket);
  void SubmitGatherPut(int entry, cudaEvent_t wait_event);
  void JoinBucket(int bucket, cudaStream_t stream) const;
  void Submit(int op, int bucket, const std::function<void()>& submit);

  // The per-op release functions; the only place the release rules live.
  // Each returns whether it released anything.
  bool ReleaseIntraCopy();
  bool ReleaseIntraSum();
  bool ReleasePut();
  bool ReleaseFinalSum(bool in_proxy);
  bool ReleaseGather(cudaEvent_t mark);
  void AfterFinalSum(int bucket, bool in_proxy);

  bool IsStepDone() const;
  void PendingCounts(int* counts) const;
  int RecordTailMark();
  void StartProxy(int slot, int intra_sum_blocks, int final_sum_blocks);
  bool ConsumePublished();
  int RunStep();
  void ProxyMain();
  void WaitOwnedGathers(double timeout_s);
  bool OwnedGatherPending(int entry, std::string* why) const;
  void ResetStep();
  void Fail(int op, int bucket, const std::string& what);
  std::string ErrorText() const;
  std::string StuckReportLocked() const;

  int device_id_{0};
  Group groups_[2];
  int64_t credit_intra_offset_{0};
  int64_t credit_inter_offset_{0};
  int64_t credit_expert_offset_{0};

  std::vector<GinBucketDesc> buckets_;
  std::vector<int> dense_order_;
  std::vector<int> expert_order_;
  // Position of each bucket in the order of its kind.
  std::vector<int> position_;
  // Gather entry of each bucket, -1 for a bucket that is not 2D.
  std::vector<int> gather_index_;
  std::vector<GatherEntry> gather_;

  cudaStream_t stream_intra_copy_{nullptr};
  cudaStream_t stream_intra_sum_{nullptr};
  cudaStream_t stream_put_{nullptr};
  cudaStream_t stream_final_sum_{nullptr};
  cudaStream_t stream_gather_put_{nullptr};
  cudaStream_t stream_gather_gate_{nullptr};
  // Per bucket, recorded by its last submission of each op.
  std::vector<cudaEvent_t> event_intra_copy_;
  std::vector<cudaEvent_t> event_intra_sum_;
  std::vector<cudaEvent_t> event_put_;
  std::vector<cudaEvent_t> event_final_sum_;
  // Per gather entry.
  std::vector<cudaEvent_t> event_gather_source_;
  std::vector<cudaEvent_t> event_gather_put_;
  std::vector<cudaEvent_t> window_marks_;
  int window_mark_index_{0};
  std::vector<cudaEvent_t> tail_marks_;
  int tail_mark_index_{0};

  // ---- accumulate across steps
  int64_t step_{0};
  // Per bucket: INTRA_COPY releases so far, the INTRA_SUM target.
  std::vector<uint64_t> copy_seq_;
  // Per bucket: puts this rank issued for it so far, the FINAL_SUM target.
  std::vector<uint64_t> targets_;
  // Per gather entry, owner only: steps waited for, the arrival target unit.
  std::vector<int64_t> gather_owner_step_;

  // ---- per-step release state (one writer, see the file header)
  // Position in the fixed order each op is released up to; dense ops index
  // dense_order_, expert ops expert_order_.
  int pos_intra_copy_{0};
  int pos_intra_sum_{0};
  int pos_dense_put_{0};
  int pos_dense_final_sum_{0};
  int pos_expert_put_{0};
  int pos_expert_final_sum_{0};
  std::vector<unsigned char> ready_;
  std::vector<float> scale_;
  cudaEvent_t current_mark_{nullptr};
  int intra_sum_blocks_{0};
  int final_sum_blocks_{0};
  int consumed_slot_{-1};
  std::vector<int> gather_pending_;
  std::vector<int> gather_put_entries_;
  // ---- per-step, caller thread only
  std::vector<unsigned char> reported_;
  int num_reported_{0};

  // ---- shared with the proxy thread, under mutex_
  mutable std::mutex mutex_;
  std::condition_variable cv_proxy_;
  std::condition_variable cv_main_;
  int state_{0};
  bool abort_{false};
  bool stop_{false};
  bool proxy_started_{false};
  int proxy_device_status_{0};
  // Per tail mark slot of this step: the bucket BucketReady() reported with
  // it, kSlotMarkOnly for the hand-over and ReleaseAsync() marks, or
  // kSlotUnused. The proxy thread takes the slots in order.
  std::vector<int> slot_bucket_;
  // Scale of a bucket reported while the proxy thread runs.
  std::vector<float> published_scale_;
  // Failure recorded by a release function.
  int err_op_{0};
  int err_bucket_{-1};
  std::string err_what_;
  // Submission in progress: op * activity scale + bucket + 1, 0 = none.
  std::atomic<int> activity_{0};

  std::thread proxy_;
};

}  // namespace distributed
}  // namespace paddle
