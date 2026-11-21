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

// release.cc: the release layer. Decides whether one stage of one bucket can
// be submitted to the GPU now and, if so, submits it through the submit_*
// functions of gin_engine.cc. Terms and constraints are in the gin_engine.h
// file header.
//
// Two releasers share the same decision code (release_next_*) and the same
// per-step state:
//   - the main thread: calls advance() in each overlap window, releases
//     whatever the stage set given by the caller allows, and returns at once.
//   - the tail thread: after the last overlap window of a step, the main
//     thread calls tail_start() to hand off to it; it keeps calling
//     release_next_* until the whole step is released.
// There is one writer at any time: the tail thread changes the per-step state
// only in RUN, advance() releases only in IDLE, and both take control->mutex
// first.
//
// After the hand-off the main thread may only call:
//   - bucket_ready: a bucket's gradients are complete, or a new mark is given.
//   - set_all_posted: every bucket of this step has been posted.
//   - wait_for_step_end: waits for the tail thread to leave RUN (on a condition
//     variable, no polling).
//   - tail_finish: cleans up after the tail thread stops; back to IDLE.
//
// Marks
//   - The main thread releases against the window mark ring (g_window_marks,
//     reused in a cycle).
//   - The tail thread uses its own tail_marks ring: each step uses the slots
//     in order from slot 0, each slot at most once per step; running out is an
//     error, there is no wrap-around. The index is cleared in step_reset.
//
// The caller (Python) decides when to advance, which stages to try in each
// window and how many blocks the reduce kernels get; this layer decides
// whether each bucket can be released and submits it. Release progress is
// available from status() / stuck_report().
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

// States of the tail thread:
//   - IDLE: the main thread releases (also between steps).
//   - RUN: the tail thread releases.
//   - DONE / ERR / ABORTED: step fully released / stopped on an error /
//     stopped on request.
// Python reads these values by name (TAIL_*).
enum { IDLE = 0, RUN = 1, DONE = 2, ERR = 3, ABORTED = 4 };
// When the tail thread has neither progress nor a new message, it waits on the
// condition variable for at most this long (microseconds) before reading the
// remote counters again. Changes of remote counters do not wake the thread, so
// they can only be polled.
const int POLL_INTERVAL_US = 10;

// Message from the main thread to the tail thread. kind: 0 = the gradients of
// dense bucket `bucket` are complete; 1 = expert bucket `bucket` is complete;
// -1 = no new bucket, only a new mark. scale = the factor applied in this
// bucket's shard reduce.
struct ReadyMessage {
  int kind, bucket, mark;
  float scale;
};
// Fixed parameters of one dense bucket. Field order matches the flat table
// that setup receives:
//   [send_offset, signal_offset, shard_numel, recv_offset_intra,
//    recv_offset_inter, intra_depth, inter_depth]
//   - *_offset: byte offset in the group's window.
//   - shard_numel: number of elements of this rank's shard.
//   - *_depth: predecessor distance, i.e. how far apart in release sequence
//     this bucket and the previous holder of the same slot of the segment are
//     (see credit_target). -1 = the bucket has no slot in this segment and
//     waits for no credit.
struct DenseBucket {
  long long send_offset, signal_offset, shard_numel, recv_offset_intra,
      recv_offset_inter, intra_depth, inter_depth;
};
// Fixed parameters of one expert bucket: [send_offset, signal_offset,
// shard_numel, recv_offset_moe, moe_depth], same meaning as DenseBucket.
struct MoeBucket {
  long long send_offset, signal_offset, shard_numel, recv_offset_moe, moe_depth;
};

// Lock, two condition variables and message queue shared by the main thread
// and the tail thread. The lock also protects the per-step release state
// below. The object is never freed after new: at process exit the tail thread
// may still wait on a cv.
struct TailControl {
  std::mutex mutex;
  // cv_proxy wakes the tail thread, cv_main wakes the main thread waiting for
  // a result
  std::condition_variable cv_proxy, cv_main;
  std::deque<ReadyMessage> queue;
  int state = IDLE;
  bool all_posted = false, abort = false, started = false;
  int set_device_rc = 0;  // return code of cudaSetDevice in the tail thread
};
TailControl* control = nullptr;
int device_id = 0;

// ---- set in setup, read-only afterwards
//   put_blocks comes from setup and has no default. The expert group size and
//   this rank's group rank are read from expert_group().
int num_dense_buckets = 0, num_moe_buckets = 0, put_blocks = 0;
// Positions of the counter areas in the host windows: intra_arrive_offset is a
// byte offset, credit_*_word is a u64 index (= g_credit_*_offset / 8). The
// intra-node arrival area and the intra-node / inter-node credits are in the
// dense group's host window, the expert credits in the expert group's.
size_t intra_arrive_offset = 0, credit_intra_word = 0, credit_inter_word = 0,
       credit_moe_word = 0;
// Fixed release order: *_release_order[position] = bucket id,
// *_pos_in_order[bucket id] = position (inverse of each other). Credit
// targets are computed from positions (see credit_target).
std::vector<int> dense_release_order, moe_release_order, dense_pos_in_order,
    moe_pos_in_order;
std::vector<DenseBucket> dense_buckets;
std::vector<MoeBucket> moe_buckets;
std::vector<cudaEvent_t> tail_marks;  // tail mark ring (see the file header)
int tail_mark_index = 0;  // slots used in this step; only the main thread
                          // changes it

// ---- per-step release state (changed by the main thread in IDLE, by the
// tail thread in RUN)
long long step = 0;  // step number used by credit targets; set by step_reset
// Position in the fixed release order each stage has been released up to, one
// variable per stage (STAGE_*). A stage's pending queue = the buckets between
// its position and the position of the previous stage.
int pos_dense_copy = 0;               // dense intra-node copy
int pos_dense_intra_node_reduce = 0;  // dense intra-node reduce
int pos_dense_inter_node_put = 0;     // dense inter-node put
int pos_dense_inter_node_reduce = 0;  // dense shard reduce
int pos_moe_put = 0;                  // expert put
int pos_moe_reduce = 0;               // expert shard reduce
// Which buckets of this step have complete gradients (bucket_ready was
// called). Cleared only in step_reset.
std::vector<unsigned char> ready_dense, ready_moe;
// Which buckets the main thread has reported through bucket_ready in this
// step. Only the main thread uses it (ready_* belongs to the tail thread
// during RUN), so a bucket reported twice is caught in both states. Cleared in
// step_reset.
std::vector<unsigned char> reported_dense, reported_moe;
// These three accumulate across steps; step_reset does not clear them:
//   - dense_copy_seq[bucket]: intra-node copies of this bucket submitted so
//     far; written into the receivers' intra-node arrival counters.
//   - dense_targets / moe_targets[bucket]: puts this rank has issued for this
//     bucket so far. The ranks are symmetric, so this is also the value this
//     rank's signal counter must reach; the shard reduce waits for it.
std::vector<unsigned long long> dense_copy_seq, dense_targets, moe_targets;
// factor applied in each bucket's shard reduce
std::vector<float> dense_scales, moe_scales;
// Block counts of the intra-node reduce / shard reduce kernels, passed by the
// caller on every advance / tail_start; no default
int intra_node_reduce_blocks = 0, shard_reduce_blocks = 0;
// Mark the next submission waits on: for the main thread, the window mark
// passed to advance; for the tail thread, the tail mark of the latest message
// (right after the hand-off, the one passed to tail_start)
cudaEvent_t current_mark = nullptr;
// Six positions and pending counts of the four stages at the moment the tail
// thread first sees "all_posted and the message queue is empty".
// posted_snapshot[0] = -1 means that moment has not come yet in this step.
// For logs only.
int posted_snapshot[6], posted_counts[4];
// Recorded on a release failure: CUDA return code / stage id / bucket id
int err_rc = 0, err_stage = 0, err_bucket = -1;
// Encoding factor of current_activity: stage id * ACTIVITY_STAGE_SCALE +
// bucket id + 1. Python reads this value by name to decode it
constexpr int ACTIVITY_STAGE_SCALE = 1000000;
// Submission in progress: stage * ACTIVITY_STAGE_SCALE + bucket id + 1;
// 0 = not inside a CUDA call
std::atomic<int> current_activity{0};

// Reads u64 number word_index of this rank's host window (acquire load, always
// from memory). Dense bucket counters live in the sharding group's host
// window, expert bucket counters in the moe_sharding group's
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

// Credit target of one recv pool segment. The bucket at position `position`
// of the fixed release order in step `step` has sequence number
// step * num_buckets + position; the previous holder of its slot has a
// sequence number smaller by `depth` (the predecessor distance). The peer
// returns one credit each time it is done with a bucket, and credits
// accumulate across steps, so target = step * num_buckets + position - depth
// + 1 (0 if negative). depth < 0 = no slot in this segment, nothing to wait
// for.
inline unsigned long long credit_target(long long num_buckets,
                                        long long position,
                                        long long depth) {
  if (depth < 0) return 0;
  const long long target = step * num_buckets + position - depth + 1;
  return target > 0 ? (unsigned long long)target : 0ull;
}

// The six blocked_* below are the remote release conditions of the stages;
// they only read counters of this rank's host window. true = cannot release
// yet. When out is non-null it receives which counter is awaited, from whom,
// its current value and its target (used by stuck_report).
struct Blocker {
  int counter;  // COUNTER_*: which counter is awaited (0 = not blocked)
  int sender;   // sender index (intra-node arrival counters only; else -1)
  int peer;     // group rank of the peer (-1 = the counter shared by all
                // senders)
  unsigned long long value, target;  // value read / target value
};
enum {
  COUNTER_INTRA_CREDIT = 1,  // intra-node credit
  COUNTER_INTRA_ARRIVE = 2,  // intra-node arrival (one sender's counter)
  COUNTER_INTER_CREDIT = 3,  // inter-node credit
  COUNTER_MOE_CREDIT = 4,    // expert credit
  COUNTER_INTER_SIGNAL = 5,  // inter-node signal (all senders combined)
  COUNTER_MOE_SIGNAL = 6     // expert signal (all senders combined)
};

// Intra-node copy can be released when the intra-node credits from every
// other rank of the node ([credit_intra + 8l]) have reached the target, i.e.
// the slot of this bucket in their intra-node segments is free
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
// Intra-node reduce can be released when, in this bucket's intra-node arrival
// counters, the counter of every other sender of the node
// ([intra_arrive + (bucket*16 + l)*8]) has reached this rank's intra-node copy
// count of this bucket
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
// Dense put can be released when the inter-node credits from every other node
// ([credit_inter + 8n]) have reached the target
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
// Expert put can be released when the expert credits from every other rank of
// the expert group ([credit_moe + 8k]) have reached the target
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
// Shard reduce can be released when this bucket's signal counter has reached
// dense_targets / moe_targets, i.e. the puts of all other ranks have arrived
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

// Whether [offset, offset + count * unit) lies inside a window of `bytes`
// bytes and offset is a multiple of align. Written so that nothing wraps
// around (offset and count come from Python).
inline bool in_window(long long offset,
                      long long count,
                      size_t unit,
                      size_t align,
                      size_t bytes) {
  if (offset < 0 || count < 0 || (size_t)offset % align) return false;
  if ((size_t)offset > bytes) return false;
  return (size_t)count <= (bytes - (size_t)offset) / unit;
}

// Checks the per-bucket tables and the host window offsets setup receives:
// every offset is non-negative, aligned (counters 8 bytes, data 4 bytes, the
// dense send_offset and recv_offset_intra 16 bytes: k_reduce_intra_node reads
// both as float4*) and inside its window, so that the CPU reads of the host
// window and the GPU writes stay in bounds. A recv offset is checked only when
// the bucket has a slot in that segment (depth >= 0).
bool tables_in_bounds(const std::vector<long long>& dense_table,
                      const std::vector<long long>& moe_table,
                      size_t intra_arrive) {
  const CommGroup& dense = dense_group();
  const CommGroup& expert = expert_group();
  const int num_dense = (int)(dense_table.size() / 7);
  const int num_moe = (int)(moe_table.size() / 5);
  const long long L = dense.ranks_per_node, Nn = dense.num_nodes;
  if (!in_window((long long)intra_arrive,
                 (long long)num_dense * 16,
                 8,
                 8,
                 dense.host_bytes) ||
      !in_window(
          (long long)g_credit_intra_offset, 16, 8, 8, dense.host_bytes) ||
      !in_window((long long)g_credit_inter_offset, Nn, 8, 8, dense.host_bytes))
    return false;
  for (int i = 0; i < num_dense; ++i) {
    const long long* row = dense_table.data() + 7 * (size_t)i;
    const long long m = row[2];
    if (m < 0 || (dense.size > 0 && m > (long long)(SIZE_MAX / 4) / dense.size))
      return false;
    if (!in_window(row[0], m * dense.size, 4, 16, dense.send_bytes) ||
        !in_window(row[1], 1, 8, 8, dense.host_bytes))
      return false;
    if (row[5] >= 0 &&
        !in_window(row[3], (L - 1) * Nn * m, 4, 16, dense.recv_bytes))
      return false;
    if (row[6] >= 0 && !in_window(row[4], Nn * m, 4, 4, dense.recv_bytes))
      return false;
  }
  if (num_moe > 0 &&
      !in_window(
          (long long)g_credit_moe_offset, expert.size, 8, 8, expert.host_bytes))
    return false;
  for (int i = 0; i < num_moe; ++i) {
    const long long* row = moe_table.data() + 5 * (size_t)i;
    const long long m = row[2];
    if (m < 0 ||
        (expert.size > 0 && m > (long long)(SIZE_MAX / 4) / expert.size))
      return false;
    if (!in_window(row[0], m * expert.size, 4, 4, expert.send_bytes) ||
        !in_window(row[1], 1, 8, 8, expert.host_bytes))
      return false;
    if (row[4] >= 0 &&
        !in_window(row[3], expert.size * m, 4, 4, expert.recv_bytes))
      return false;
  }
  return true;
}

// Records where a failure happened and returns -1; the caller then stops
// releasing for this step
int record_failure(int rc, int stage, int bucket) {
  err_rc = rc;
  err_stage = stage;
  err_bucket = bucket;
  return -1;
}

// ---- Muon 2D-bucket gather. After the tail thread releases the shard reduce
// of a 2D bucket, it sends this rank's shard to the owner right away. Shard
// reduces released by the main thread do not send the gather put here; the
// caller sends it in a window through gather_put_entry.
//   - The table is built by setup_gather_table; without it gather_enabled = 0
//     and this path is not taken.
//   - One entry per 2D bucket: kind, bucket, slot (event index), whether this
//     rank is the owner, the owner's group rank, the offset send_offset and
//     element count shard_numel of this rank's shard in the send window, and
//     the arrival counter in the owner's host window (arrive_offset across
//     nodes, intra_arrive_offset within the node).
//   - gather_seq[entry]: put sequence number of this shard in this step (the
//     value written into the arrival counter within a node), set by
//     set_gather_sequences before the hand-off. 0 = the tail thread does not
//     send it in this step.
//   - gather_sent / gather_sent_seq: entries the tail thread sent in this step
//     and the sequence numbers used, read by get_gather_sent.
//   - Like the other per-step state, the main thread may read or write these
//     only while the tail thread is not in RUN.
struct GatherEntry {
  int kind, bucket, slot, is_owner, owner_rank;
  long long send_offset, shard_numel;
  size_t arrive_offset, intra_arrive_offset;
};
std::vector<GatherEntry> gather_entries;
// bucket id -> table index, -1 = not a 2D bucket
std::vector<int> gather_index_dense, gather_index_moe;
std::vector<unsigned long long> gather_seq;
std::vector<int> gather_sent;
std::vector<unsigned long long> gather_sent_seq;
int gather_enabled = 0;

// Called by the tail thread after the shard reduce of a bucket is submitted to
// the GPU. Does nothing for a bucket that is not 2D or when this rank is the
// owner; otherwise makes g_stream_gather_put wait on this bucket's shard
// reduce event and calls gather_put_to_owner, without waiting on remote
// conditions. Returns 0 or a CUDA error code.
int send_gather_put_after_shard_reduce(int kind, int bucket) {
  const int entry_index =
      (kind ? gather_index_moe : gather_index_dense)[bucket];
  if (entry_index < 0) return 0;
  const GatherEntry& entry = gather_entries[entry_index];
  if (entry.is_owner) return 0;
  const unsigned long long seq = gather_seq[entry_index];
  if (seq == 0)
    return (int)cudaErrorInvalidValue;  // no sequence number in this step, or
                                        // already sent in this step
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
  // check the signal count: none within the node (CE), one per put across
  // nodes
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

// Each of the four release_next_* handles one stage: starting at the next
// unreleased bucket of the fixed release order, release every bucket that can
// go and stop at the first one that cannot (later buckets are never released
// ahead of it). Both the main thread and the tail thread call these; they
// differ only in current_mark and the block counts.
// Returns 1 = released at least one, 0 = none, -1 = a CUDA call failed
// (recorded in err_*).
int release_next_intra_node_copy() {
  int moved = 0;
  while (pos_dense_copy < num_dense_buckets) {
    const int bucket = dense_release_order[pos_dense_copy];
    if (!ready_dense[bucket] || blocked_intra_node_copy(bucket, nullptr))
      break;  // gradients not complete yet, or the receivers' slot is not free
    const DenseBucket& bucket_params = dense_buckets[bucket];
    // this rank's counter in each receiver's intra-node arrival area (same
    // offset on every receiver)
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
  while (pos_dense_intra_node_reduce < num_dense_buckets) {
    const int bucket = dense_release_order[pos_dense_intra_node_reduce];
    // requires this bucket's intra-node copy to be released and the rows
    // pushed by the other ranks to have arrived
    if (pos_dense_intra_node_reduce >= pos_dense_copy ||
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
    ++pos_dense_intra_node_reduce;
    moved = 1;
  }
  return moved;
}
int release_next_put() {
  int moved = 0, num_signals = 0;
  while (pos_dense_inter_node_put < num_dense_buckets) {
    const int bucket = dense_release_order[pos_dense_inter_node_put];
    // requires this bucket's intra-node reduce to be released and the slot in
    // the peers' inter-node segment to be free
    if (pos_dense_inter_node_put >= pos_dense_intra_node_reduce ||
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
    ++pos_dense_inter_node_put;
    moved = 1;
  }
  while (pos_moe_put < num_moe_buckets) {
    const int bucket = moe_release_order[pos_moe_put];
    if (!ready_moe[bucket] || blocked_moe_put(bucket, nullptr))
      break;  // gradients not complete yet, or the peers' slot is not free
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
// The shard reduce takes two extra arguments:
//   - in_tail = 1 (called by the tail thread): after releasing the shard
//     reduce of a 2D bucket, send its gather put right away
//     (send_gather_put_after_shard_reduce).
//   - released non-null (called by the main thread): appends, in release
//     order, the shard reduces released by this call, two numbers each
//     [kind, bucket id].
int release_next_shard_reduce(int in_tail, std::vector<int>* released) {
  int moved = 0;
  while (pos_dense_inter_node_reduce < num_dense_buckets) {
    const int bucket = dense_release_order[pos_dense_inter_node_reduce];
    // requires this bucket's put to be released and the puts of all other
    // ranks to have arrived
    if (pos_dense_inter_node_reduce >= pos_dense_inter_node_put ||
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
    ++pos_dense_inter_node_reduce;
    if (released) {
      released->push_back(0);  // kind 0 = dense
      released->push_back(bucket);
    }
    if (in_tail && gather_enabled) {  // 1 only when a 2D-bucket table exists
      current_activity = STAGE_GATHER_DENSE * ACTIVITY_STAGE_SCALE + bucket + 1;
      const int gather_rc = send_gather_put_after_shard_reduce(0, bucket);
      current_activity = 0;
      if (gather_rc)
        return record_failure(gather_rc, STAGE_GATHER_DENSE, bucket);
    }
    moved = 1;
  }
  while (pos_moe_reduce < num_moe_buckets) {
    const int bucket = moe_release_order[pos_moe_reduce];
    if (pos_moe_reduce >= pos_moe_put || blocked_moe_reduce(bucket, nullptr))
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
    ++pos_moe_reduce;
    if (released) {
      released->push_back(1);  // kind 1 = expert
      released->push_back(bucket);
    }
    if (in_tail && gather_enabled) {  // expert 2D bucket, as above
      current_activity = STAGE_GATHER_MOE * ACTIVITY_STAGE_SCALE + bucket + 1;
      const int gather_rc = send_gather_put_after_shard_reduce(1, bucket);
      current_activity = 0;
      if (gather_rc) return record_failure(gather_rc, STAGE_GATHER_MOE, bucket);
    }
    moved = 1;
  }
  return moved;
}
// One round of the tail thread: try shard reduce -> intra-node copy ->
// intra-node reduce -> put once each. Shard reduce goes first because it frees
// recv pool slots. Returns 1 = progress, 0 = none, -1 = error.
// The main thread does not call this function.
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
// How many buckets of each of the four stages have reached it but are not
// released yet, in the order [intra-node copy, intra-node reduce, put, shard
// reduce] (put and shard reduce count dense and expert). Intra-node copy and
// expert put count only buckets that are ready. For logs only.
void pending_counts(int* out4) {
  int ready_copy = 0, ready_moe_put = 0;
  for (int k = pos_dense_copy; k < num_dense_buckets; ++k)
    if (ready_dense[dense_release_order[k]]) ++ready_copy;
  for (int k = pos_moe_put; k < num_moe_buckets; ++k)
    if (ready_moe[moe_release_order[k]]) ++ready_moe_put;
  out4[0] = ready_copy;
  out4[1] = pos_dense_copy - pos_dense_intra_node_reduce;
  out4[2] =
      pos_dense_intra_node_reduce - pos_dense_inter_node_put + ready_moe_put;
  out4[3] = pos_dense_inter_node_put - pos_dense_inter_node_reduce +
            pos_moe_put - pos_moe_reduce;
}
// Whether the whole step is released: all six positions are at the end
bool is_step_done() {
  return pos_dense_copy >= num_dense_buckets &&
         pos_dense_intra_node_reduce >= num_dense_buckets &&
         pos_dense_inter_node_put >= num_dense_buckets &&
         pos_dense_inter_node_reduce >= num_dense_buckets &&
         pos_moe_put >= num_moe_buckets && pos_moe_reduce >= num_moe_buckets;
}

// Main loop of one tail-thread step (in RUN only this thread changes the
// per-step state): take one message -> call release_pass_once until it makes
// no progress -> return DONE once everything is released. With neither a
// message nor progress, wait on the condition variable for at most
// POLL_INTERVAL_US (a new message, all_posted and abort wake it). CUDA calls
// must be made outside the lock, or they would block the main thread from
// sending messages.
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
      // record the six positions and pending counts at the moment all_posted
      // is set and the queue is empty (for logs)
      const int snapshot[6] = {pos_dense_copy,
                               pos_dense_intra_node_reduce,
                               pos_dense_inter_node_put,
                               pos_dense_inter_node_reduce,
                               pos_moe_put,
                               pos_moe_reduce};
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
      cudaSetDevice(device_id);  // same device as the main thread (primary
                                 // context)
  // timer slack of this thread = 1 us, so the short POLL_INTERVAL_US waits
  // wake up on time
  prctl(PR_SET_TIMERSLACK, 1000UL, 0UL, 0UL, 0UL);
  std::unique_lock<std::mutex> lk(control->mutex);
  control->set_device_rc = (int)device_err;
  control->started = true;
  control->cv_main.notify_all();
  if (device_err != cudaSuccess) return;
  for (;;) {
    control->cv_proxy.wait(lk, [] {
      return control->state == RUN;
    });  // between steps, wait on the condition variable, no polling
    lk.unlock();
    const int state = run_step();
    lk.lock();
    control->state = state;
    control->cv_main.notify_all();
  }
}

}  // namespace

// Init, called once after configure: takes the fixed release orders and the
// per-bucket parameter tables, creates the tail mark ring (num_marks slots)
// and starts the tail thread (which then waits in IDLE).
//   - Bucket counts come from the sizes of g_event_dense_put /
//     g_event_moe_put; dense_order / moe_order must be permutations of the
//     bucket ids and the table sizes must match the bucket counts, otherwise
//     an error is returned.
//   - dense_table has 7 numbers per bucket, moe_table 5 (fields as in
//     DenseBucket / MoeBucket).
//   - intra_arrive_offset_: byte offset of the intra-node arrival area in the
//     dense group's host window.
//   - The positions of the credit areas come from g_credit_*_offset.
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
  if (!tables_in_bounds(dense_table, moe_table, intra_arrive_offset_))
    return (int)cudaErrorInvalidValue;
  CUDA_CHECK(cudaGetDevice(&device_id));
  num_dense_buckets = num_dense;
  num_moe_buckets = num_moe;
  put_blocks = put_blocks_;
  intra_arrive_offset = intra_arrive_offset_;
  credit_intra_word = g_credit_intra_offset / 8;  // byte offset -> u64 index
  credit_inter_word = g_credit_inter_offset / 8;
  credit_moe_word = g_credit_moe_offset / 8;
  dense_release_order = dense_order;
  moe_release_order = moe_order;
  dense_pos_in_order.assign(num_dense_buckets, -1);
  moe_pos_in_order.assign(num_moe_buckets, -1);
  // build "bucket id -> position" and check that the order is a permutation
  // of the bucket ids
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
  reported_dense.assign(num_dense_buckets, 0);
  reported_moe.assign(num_moe_buckets, 0);
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
    // no thread: unpublish control so that setup can be retried
    delete control;
    control = nullptr;
    return (int)cudaErrorUnknown;
  }
  std::unique_lock<std::mutex> lk(control->mutex);
  control->cv_main.wait(lk, [] { return control->started; });
  return control->set_device_rc;
}

// Main thread records a tail mark at the current position of `stream` and
// returns its slot; -1 = not set up or cudaEventRecord failed, -2 = the slots
// of this step are used up
int record_tail_mark(cudaStream_t stream) {
  if (!control) return -1;
  if (tail_mark_index >= (int)tail_marks.size()) return -2;
  const int i = tail_mark_index;
  if (cudaEventRecord(tail_marks[i], stream) != cudaSuccess) return -1;
  ++tail_mark_index;
  return i;
}

// Called before each step: clears the per-step state and sets the step number
// step_ (= number of finished steps, absolute). step is used only by credit
// targets (see credit_target). dense_copy_seq / dense_targets / moe_targets
// accumulate across steps and are not cleared. Only callable in IDLE.
int step_reset(long long step_) {
  if (!control) return (int)cudaErrorNotReady;
  std::lock_guard<std::mutex> lk(control->mutex);
  if (control->state != IDLE) return (int)cudaErrorInvalidValue;
  step = step_;
  pos_dense_copy = 0;
  pos_moe_put = 0;
  pos_dense_intra_node_reduce = pos_dense_inter_node_put = 0;
  pos_dense_inter_node_reduce = pos_moe_reduce = 0;
  std::fill(ready_dense.begin(), ready_dense.end(), 0);
  std::fill(ready_moe.begin(), ready_moe.end(), 0);
  std::fill(reported_dense.begin(), reported_dense.end(), 0);
  std::fill(reported_moe.begin(), reported_moe.end(), 0);
  for (int& position : posted_snapshot) position = -1;
  for (int& count : posted_counts) count = 0;
  err_rc = 0;
  err_stage = 0;
  err_bucket = -1;
  current_mark = nullptr;
  tail_mark_index = 0;  // the tail mark ring restarts at slot 0 every step
  return 0;
}

// Main thread releases inside one overlap window.
//   - stages: the stages to try, ADVANCE_* OR-ed together.
//   - mark_event: the window mark of this window.
//   - the two block counts: blocks of the intra-node reduce / shard reduce
//     kernels submitted by this call.
//   - The selected stages are tried once each in the order intra-node copy ->
//     intra-node reduce -> put -> shard reduce, without waiting on remote
//     counters; what cannot go now is left to a later window or the tail
//     thread.
//   - released_shard_reduce: the shard reduces released by this call, in
//     release order, two numbers each [kind, bucket id].
//   - Returns 0 or a CUDA error code (where it failed: get_last_error). Only
//     callable in IDLE; otherwise returns cudaErrorInvalidValue.
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

// Hands release off to the tail thread: state IDLE -> RUN. Only callable in
// IDLE.
//   - the two block counts: blocks of the intra-node reduce / shard reduce
//     kernels the tail thread submits.
//   - mark_: slot of the tail mark recorded at the hand-off.
//   - all_posted = 1: every bucket of this step was posted before the
//     hand-off (the step never reached the tail; the hand-off happens in
//     reduce_gradients).
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

// A bucket's gradients are complete (kind 0 = dense bucket, 1 = expert
// bucket), or only a new mark is given (kind -1). scale = the factor applied
// in this bucket's shard reduce.
//   - IDLE: mark_ must be -1; ready and scale are recorded directly and a
//     later advance releases the bucket.
//   - RUN: mark_ is a tail mark slot; this is sent to the tail thread as a
//     message.
//   - Returns 0 = accepted; 1 = the tail thread stopped on an error (reason:
//     get_last_error); 2 = usage error (mark does not match the state, invalid
//     bucket id, bucket already reported in this step (checked in both
//     states), a bucket becomes ready after everything was released).
int bucket_ready(int kind, int bucket, float scale, int mark_) {
  if (!control) return 2;
  std::lock_guard<std::mutex> lk(control->mutex);
  if (control->state == ERR) return 1;
  if (kind > 1 ||
      (kind >= 0 &&
       (bucket < 0 || bucket >= (kind ? num_moe_buckets : num_dense_buckets))))
    return 2;
  if (control->state == IDLE) {  // main thread still releasing: record now
    if (mark_ >= 0 || kind < 0 || is_step_done()) return 2;
    unsigned char& reported = (kind ? reported_moe : reported_dense)[bucket];
    unsigned char& ready =
        (kind ? ready_moe : ready_dense)[bucket];  // ready once per step
    if (reported || ready) return 2;
    reported = 1;
    ready = 1;
    (kind ? moe_scales : dense_scales)[bucket] = scale;
    return 0;
  }
  if (mark_ < 0 || mark_ >= tail_mark_index) return 2;
  if (control->state == DONE)
    return kind < 0 ? 0 : 2;  // all released: drop mark-only messages
  if (control->state != RUN) return 2;
  if (kind >= 0) {
    unsigned char& reported = (kind ? reported_moe : reported_dense)[bucket];
    if (reported) return 2;  // reported twice in this step
    reported = 1;
  }
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

// Waits up to timeout_s seconds for the tail thread to leave RUN (released /
// error / stopped) and returns the state at that point; on timeout it returns
// RUN. Waits on a condition variable, no polling.
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

// Asks the tail thread to stop, waits up to wait_s seconds and returns the
// state afterwards. The tail thread checks abort only at the start of each
// round; if it is stuck inside a CUDA call, this waits until the timeout and
// returns RUN
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

// Cleanup after the tail thread stops: clears the message queue and the
// all_posted / abort flags, back to IDLE. Only callable in DONE / ERR /
// ABORTED.
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

// Release progress of this step. Not callable in RUN.
//   - positions6 = positions released up to by [dense intra-node copy, dense
//     intra-node reduce, dense put, dense shard reduce, expert put, expert
//     shard reduce].
//   - counts4 = pending counts of the four stages (order as in
//     pending_counts).
//   - posted4 = the same numbers at the posted_snapshot moment; [-1, 0, 0, 0]
//     if that moment has not come.
//   - *done = whether the whole step is released.
int status(std::vector<int>& positions6,
           std::vector<int>& counts4,
           std::vector<int>& posted4,
           int* done) {
  if (!control) return (int)cudaErrorNotReady;
  std::lock_guard<std::mutex> lk(control->mutex);
  if (control->state == RUN) return (int)cudaErrorNotReady;
  positions6 = {pos_dense_copy,
                pos_dense_intra_node_reduce,
                pos_dense_inter_node_put,
                pos_dense_inter_node_reduce,
                pos_moe_put,
                pos_moe_reduce};
  int counts[4] = {0, 0, 0, 0};
  pending_counts(counts);
  counts4.assign(counts, counts + 4);
  posted4.assign(4, 0);
  posted4[0] = -1;
  if (posted_snapshot[0] >= 0) posted4.assign(posted_counts, posted_counts + 4);
  *done = is_step_done() ? 1 : 0;
  return 0;
}

// Snapshot of a stuck step (only reads the host window, changes no state).
// Not callable in RUN.
// flat holds one record per stage, six in all (the four dense stages, then
// the two expert stages). Each record is:
//   - a 6-number header: [stage id STAGE_*, position, number of buckets,
//     whether the next bucket has reached this stage (1 / 0), pending count,
//     number n of buckets listed after it].
//   - n groups of 6 numbers (n <= STUCK_MAX_ITEMS): [bucket id, awaited
//     counter COUNTER_* (0 = condition met, only queued behind the previous
//     bucket), sender index (-1 = not applicable), peer group rank (-1 = the
//     counter shared by all senders), current value, target value].
const int STUCK_MAX_ITEMS = 8;

static void stuck_stage(int stage, std::vector<long long>& flat) {
  const bool is_moe = stage == STAGE_MOE_PUT || stage == STAGE_MOE_REDUCE;
  const std::vector<int>& order =
      is_moe ? moe_release_order : dense_release_order;
  const int num_buckets = is_moe ? num_moe_buckets : num_dense_buckets;
  // previous = position of the previous stage (-1 = use ready)
  int position = 0, previous = 0;
  switch (stage) {
    case STAGE_DENSE_INTRA_NODE_COPY:
      position = pos_dense_copy;
      previous = -1;
      break;
    case STAGE_DENSE_INTRA_NODE_REDUCE:
      position = pos_dense_intra_node_reduce;
      previous = pos_dense_copy;
      break;
    case STAGE_DENSE_INTER_NODE_PUT:
      position = pos_dense_inter_node_put;
      previous = pos_dense_intra_node_reduce;
      break;
    case STAGE_DENSE_INTER_NODE_REDUCE:
      position = pos_dense_inter_node_reduce;
      previous = pos_dense_inter_node_put;
      break;
    case STAGE_MOE_PUT:
      position = pos_moe_put;
      previous = -1;
      break;
    default:  // STAGE_MOE_REDUCE
      position = pos_moe_reduce;
      previous = pos_moe_put;
      break;
  }
  // pending queue of this stage: buckets after the position that have reached
  // this stage (ready, or the previous stage's position is past them)
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

// out3 = [CUDA return code, stage id (STAGE_*), bucket id]; [0, 0, -1] when no
// error occurred
int get_last_error(int* out3) {
  if (!control) return (int)cudaErrorNotReady;
  std::lock_guard<std::mutex> lk(control->mutex);
  out3[0] = err_rc;
  out3[1] = err_stage;
  out3[2] = err_bucket;
  return 0;
}

// The submission in progress, encoded as current_activity; 0 = not inside a
// CUDA call
int get_current_activity() { return current_activity.load(); }

// Constant table exported to Python (name -> value): ADVANCE_*, STAGE_*,
// COUNTER_*, TAIL_* and ACTIVITY_STAGE_SCALE. gin_overlap_api.cc exports them
// as module attributes of paddle.base.core.gin_overlap. Adding, changing or
// removing a constant requires changing this table too.
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

// Builds the 2D-bucket table. Call it after setup and gather_setup, in IDLE,
// only once. Each entry of table has 10 numbers:
//   [kind (0 dense / 1 expert), bucket id, slot, whether this rank is the
//    owner, owner's group rank, send_offset, shard_numel, reserved (must be
//    >= 1), arrive_offset, intra_arrive_offset]
// An empty table builds nothing and gather_enabled stays 0.
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
    // the two arrival counters: non-negative, 8-byte aligned, inside the
    // group's host window (all ranks share the host window layout)
    const size_t host_bytes =
        group_at(entry.kind ? GROUP_MOE : GROUP_DENSE).host_bytes;
    if (!in_window(row[8], 1, 8, 8, host_bytes) ||
        !in_window(row[9], 1, 8, 8, host_bytes))
      return (int)cudaErrorInvalidValue;
    int& entry_index = (entry.kind ? index_moe : index_dense)[entry.bucket];
    if (entry_index >= 0)
      return (int)cudaErrorInvalidValue;  // two entries for the same bucket
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

// Main thread sends the gather put of one 2D bucket: looks up the entry built
// by setup_gather_table and waits on g_event_gather_source[slot] (recorded by
// gather_mark_ready). Only callable in IDLE.
//   - seq: put sequence number of this shard in this step (written into the
//     owner's arrival counter within a node); must be > 0.
//   - *num_signals: number of signals sent (0 within the node, 1 across
//     nodes).
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

// Sets this step's put sequence number of every entry (0 = the tail thread
// does not send it in this step) and clears the previous step's send record.
// Call it before tail_start, in IDLE
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

// Gather puts the tail thread sent in this step, in send order, three numbers
// each [kind, bucket, seq]. Not callable in RUN
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
