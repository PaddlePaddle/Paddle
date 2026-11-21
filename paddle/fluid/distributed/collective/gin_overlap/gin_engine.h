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

// gin_engine.h: everything gin_overlap declares. Global state (windows, side
// streams, events, offsets), error-check macros, the per-stage submit
// functions, the release layer interface, and GradCommEngine, the class
// exported to Python.
//
// gin_overlap splits the reduce_scatter of each sharding-stage1 gradient
// bucket into stages built on NCCL GIN one-sided puts. Each stage is submitted
// to the GPU on its own during the backward pass. This header is the complete
// description of the C++ side; other files refer to it.
//
// Terms
//   - bucket: a contiguous block of device memory holding the gradients of
//     several parameters; communication works on whole buckets.
//   - shard: the part of a bucket this rank owns after the bucket is split
//     across the communication group.
//   - GIN: NCCL GPU-Initiated Networking. Only its one-sided put is used: it
//     writes a range of local data into a peer's window without involving the
//     peer GPU.
//   - LSA: in the NCCL device API, the ranks of one node that can reach each
//     other's memory through plain pointers (Load/Store Accessible). lsaRank /
//     lsaSize are this rank's index in the node / the number of ranks in it.
//   - devcomm: ncclDevComm, the communicator handle that kernels use for GIN
//     and LSA calls.
//   - CE: copy engine. cudaMemcpyBatchAsync runs on it and takes no SMs.
//   - window: symmetric memory registered with NCCL that peers can read and
//     write one-sidedly. Each communication group has three device windows
//     (send / recv / signal) and one host window (CPU memory that holds the
//     counters). The signal window only reserves the offsets: counters are
//     written to the same offset of the host window. A bucket's grad_storage
//     is its range of the send window (send_window_tensor wraps it as a
//     tensor without copying).
//   - overlap window: a span of the backward pass in which DeepEP leaves the
//     NIC or the SMs idle. Window A / B and the pre window below are overlap
//     windows, and so is the "window" in record_window_mark.
//   - mark: a cudaEvent recorded on the compute stream. Work submitted to the
//     GPU waits on a mark first, so it cannot start earlier than that point
//     and the gradients it reads are already written.
//   - release: deciding that one stage of one bucket may run now, and
//     submitting it to the GPU. The release layer (release.cc) makes that
//     decision.
//   - credit: after a receiver is done with a slot of its recv pool, it adds
//     1 to a counter of the sender ("returns a credit"). The sender writes to
//     that slot again only once the counter reaches its target.
//   - owner: the rank a Muon 2D bucket finally belongs to.
//   - group rank: a rank's index inside a communication group, not its world
//     rank.
//   - tail thread: the background thread of the release layer. It releases
//     everything that is left after the last overlap window of a step.
//
// Communication groups
//   - Dense buckets use the sharding group (GROUP_DENSE), expert buckets the
//     moe_sharding group (GROUP_MOE). Each group has its own ncclComm, devcomm
//     and windows.
//   - Ranks of a group must be laid out node-major:
//     rank_in_node = rank % ranks_per_node, node_index = rank / ranks_per_node.
//
// The four stages of a dense bucket. Rank (node_index, rank_in_node) owns
// column rank_in_node, i.e. the blocks
// {n * ranks_per_node + rank_in_node | n = 0..num_nodes-1}:
//   1. intra-node copy: the CE copies this rank's blocks of column l into this
//      rank's row of the intra-node segment of rank l's recv window (l is a
//      rank of the same node), then writes this rank's intra-node arrival
//      counter in rank l's host window.
//   2. intra-node reduce: each block of this rank's column becomes the sum of
//      the matching blocks of all ranks of the node, written back in place
//      into grad; it is the source of the put. Then an intra-node credit is
//      returned to the other ranks of the node.
//   3. inter-node put: block n of this column goes to the rank with the same
//      rank_in_node on node n, into row node_index of its inter-node segment;
//      the put also adds 1 to that rank's signal counter of this bucket in
//      its host window.
//   4. shard reduce: this rank's shard = (sum of the partial sums of all
//      nodes) * scale. Then an inter-node credit is returned to the ranks with
//      the same rank_in_node on the other nodes.
// An expert bucket has two stages: the expert put to every other rank of the
// expert group, then the shard reduce and an expert credit.
// A Muon 2D bucket additionally gathers this rank's shard to the owner after
// the shard reduce (GIN put across nodes, CE + release store within a node).
//
// In names, the shard reduce is inter_node_reduce for dense buckets and
// moe_reduce for expert buckets; shard_reduce covers both (for example
// g_stream_shard_reduce, ADVANCE_SHARD_REDUCE).
//
// Recv pool (the recv window)
//   - The dense group's recv window is [intra-node segment][inter-node
//     segment]; the expert group's is [expert segment].
//   - Each segment is a ring: buckets take one slot each in the fixed release
//     order and wrap to the start of the segment when the rest does not fit,
//     so several buckets of one segment can be in flight at once.
//   - Before a new bucket reuses a slot, the receiver must be done with the
//     bucket that held it and have returned the credit.
//
// Host window layout (one HOST_NUMA allocation per group, one u64 per
// counter; Python computes the offsets and passes them in)
//   - dense group:
//     - [signal_offset]: one counter per dense bucket, the GIN signal carried
//       by the inter-node put.
//     - [intra_arrive + (bucket*16 + l)*8]: intra-node arrival counters.
//       Sender l of the same node writes how many intra-node copies of this
//       bucket it has submitted.
//     - [credit_intra + 8*l]: +1 each time rank l of the node finishes an
//       intra-node reduce.
//     - [credit_inter + 8*n]: +1 each time the rank with the same
//       rank_in_node on node n finishes a shard reduce.
//     - arrival counters of the 2D-bucket gather.
//   - expert group:
//     - [signal_offset]: one counter per expert bucket, the GIN signal carried
//       by the expert put.
//     - [credit_moe + 8*k]: +1 each time group rank k finishes a shard reduce.
//     - arrival counters of the 2D-bucket gather.
//
// Constraints
//   - Counters accumulate across steps and only grow. The CPU never writes or
//     clears them.
//   - A u64 is written either by the NIC (atomics) or by the GPU, never both:
//     signal counters and inter-node / expert credits are written only by the
//     NIC, intra-node arrival counters and intra-node credits only by the GPU.
//   - Each sender writes only its own intra-node arrival counter; a counter
//     has exactly one writer.
//   - Every stage is submitted in the fixed release order, which must be the
//     same on all ranks; credit targets are computed from that order.
//   - Work submitted to the GPU waits only on local events (marks on the
//     compute stream, events of local side streams), never on remote
//     counters. The release layer reads remote counters from the host window
//     on the CPU.
//   - Every bucket goes through all of its stages every step; otherwise later
//     credits never arrive.
//   - The release state has one writer at any time: the main thread (advance)
//     or the tail thread.
//   - The engine lives as long as the process. Its windows are registered on
//     the ncclComm of Paddle's ProcessGroup and are never deregistered or
//     freed; the devcomms, streams and events are never destroyed either.
//     shutdown_process_group / restart_process_group are not supported: using
//     the engine after either of them is undefined behavior.
//   - An init step that fails can be retried only if its comment says so
//     (set_comms, set_group_topology, configure); otherwise restart the
//     process.
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <nccl_device.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "paddle/fluid/distributed/collective/gin_overlap/gin_dynload.h"
#include "paddle/fluid/distributed/collective/gin_overlap/kernels/kernels.h"

namespace gin_overlap {

// Return codes: 0 = success; < 10000 = cudaError_t; >= 10000 = 10000 +
// ncclResult_t.
#define CUDA_CHECK(x)                                       \
  do {                                                      \
    cudaError_t _err = (x);                                 \
    if (_err != cudaSuccess) return static_cast<int>(_err); \
  } while (0)
#define NCCL_CHECK(x)                                               \
  do {                                                              \
    ncclResult_t _res = (x);                                        \
    if (_res != ncclSuccess) return 10000 + static_cast<int>(_res); \
  } while (0)
// Calls something that already returns a code in this convention (a launch_*
// wrapper, a submit_* function, ...) and returns that code on failure.
#define GIN_CHECK(x)                     \
  do {                                   \
    const int _rc = static_cast<int>(x); \
    if (_rc) return _rc;                 \
  } while (0)
// Driver API failure: prints the CUresult, then returns cudaErrorUnknown
// (CUresult and cudaError_t use different numbering).
#define CUDA_DRIVER_CHECK(x)                      \
  do {                                            \
    CUresult _res = (x);                          \
    if (_res != CUDA_SUCCESS) {                   \
      const char* _msg = nullptr;                 \
      cuGetErrorString(_res, &_msg);              \
      fprintf(stderr,                             \
              "[gin_overlap] %s failed: %d %s\n", \
              #x,                                 \
              static_cast<int>(_res),             \
              _msg ? _msg : "?");                 \
      return static_cast<int>(cudaErrorUnknown);  \
    }                                             \
  } while (0)

// ----------------------------------------------------------------
// Communication groups: one ncclComm, devcomm and window set per group (see
// the file header). Addressing inside a group always uses group ranks.
enum { GROUP_DENSE = 0, GROUP_MOE = 1, NUM_GROUPS = 2 };

struct CommGroup {
  ncclComm_t comm = nullptr;
  ncclDevComm_t dev_comm{};
  // GIN contexts the devcomm actually provides (the put kernels pick one by
  // blockIdx)
  int num_gin_contexts = 0;
  int size = 0, rank = 0;  // group size, this rank's group rank
  int ranks_per_node = 1;  // group members per node (= dev_comm.lsaSize)
  int rank_in_node = 0;    // this rank's index in the node (= dev_comm.lsaRank)
  int num_nodes = 1;       // nodes in the group
  int node_index = 0;      // index of this rank's node
  ncclWindow_t win_send{}, win_recv{}, win_signal{}, win_host{};
  char* send = nullptr;  // send window: grad_storage of this group's buckets
  // dense group = [intra-node segment][inter-node segment];
  // expert group = [expert segment]
  char* recv = nullptr;
  char* signal = nullptr;  // offsets only: counters live in the host window
  char* host = nullptr;    // CPU address of the host window (read by release)
  // actual sizes of the send / recv / host windows
  size_t send_bytes = 0, recv_bytes = 0, host_bytes = 0;
  CUmemGenericAllocationHandle host_handle = 0;
  // start of the recv / signal / host / send window of node-local rank l
  char* lsa_recv[16] = {};
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
// Global state (defined in gin_engine.cc)
extern int g_device_id;  // device of this process (set in set_comms)
extern cudaStream_t g_stream_intra_node_copy;    // intra-node copy
extern cudaStream_t g_stream_intra_node_reduce;  // intra-node reduce
extern cudaStream_t g_stream_put;                // dense put + expert put
extern cudaStream_t g_stream_shard_reduce;       // dense + expert shard reduce
extern std::vector<cudaEvent_t> g_event_intra_node_copy,
    g_event_intra_node_reduce, g_event_dense_put, g_event_moe_put;
extern std::vector<cudaEvent_t> g_event_dense_inter_node_reduce,
    g_event_moe_reduce;
extern std::vector<cudaEvent_t> g_window_marks;  // window mark ring (reused)
extern int g_window_mark_index;
extern uint64_t g_inter_node_bytes, g_intra_node_bytes;
// Byte offsets of the credit areas in the host windows: intra-node /
// inter-node in the dense group's host window, expert in the expert group's
extern size_t g_credit_intra_offset, g_credit_inter_offset, g_credit_moe_offset;
extern int g_buckets_ready;
// 2D-bucket gather: the sender's put / CE
extern cudaStream_t g_stream_gather_put;
// Gate stream of the 2D-bucket gather (created in gather_setup): only
// cudaStreamWaitEvent and cudaEventRecord go on it, no kernels
extern cudaStream_t g_stream_gather_gate;
// one per slot: after this rank's shard reduce
extern std::vector<cudaEvent_t> g_event_gather_source;
// one per slot: after this rank's gather put / CE
extern std::vector<cudaEvent_t> g_event_gather_put;
extern int g_gather_ready;

// ----------------------------------------------------------------
// Paddle's current compute stream (implemented in paddle_tensor.cc), the same
// stream as Python's paddle.device.current_stream(). On failure it prints the
// reason and returns an error code; it never falls back to the default stream.
int paddle_compute_stream(cudaStream_t* out);
// Gets the current compute stream; on failure the enclosing method returns
// the error code
#define GIN_COMPUTE_STREAM(var)                         \
  cudaStream_t var = nullptr;                           \
  do {                                                  \
    const int _stream_rc = paddle_compute_stream(&var); \
    if (_stream_rc) return _stream_rc;                  \
  } while (0)

// ----------------------------------------------------------------
// Per-stage submit functions: submit one stage of one bucket to the GPU
// (launch a kernel or a CE copy, record an event). Only release_next_* in
// release.cc calls them, after the remote conditions are met. mark_event is
// the mark this work waits on first (the main thread passes a window mark,
// the tail thread a tail mark).
int submit_dense_intra_node_copy(int bucket,
                                 size_t recv_offset,
                                 size_t send_offset,
                                 int64_t shard_numel,
                                 cudaEvent_t mark_event,
                                 size_t intra_arrive_offset,
                                 uint64_t seq);
int submit_dense_intra_node_reduce(int bucket,
                                   size_t recv_offset,
                                   size_t send_offset,
                                   int64_t shard_numel,
                                   int num_blocks,
                                   cudaEvent_t mark_event);
int submit_dense_inter_node_put(int bucket,
                                size_t recv_offset,
                                size_t send_offset,
                                size_t signal_offset,
                                int64_t shard_numel,
                                int num_blocks,
                                cudaEvent_t mark_event,
                                int* num_signals);
// The two stages of an expert bucket: group size and this rank's group rank
// come from expert_group()
int submit_moe_put(int bucket,
                   size_t recv_offset,
                   size_t send_offset,
                   size_t signal_offset,
                   int64_t shard_numel,
                   int num_blocks,
                   cudaEvent_t mark_event,
                   int* num_signals);
int submit_dense_inter_node_reduce(int bucket,
                                   size_t recv_offset,
                                   size_t send_offset,
                                   int64_t shard_numel,
                                   float scale,
                                   int num_blocks,
                                   cudaEvent_t mark_event);
int submit_moe_reduce(int bucket,
                      size_t recv_offset,
                      size_t send_offset,
                      int64_t shard_numel,
                      float scale,
                      int num_blocks,
                      cudaEvent_t mark_event);
// Send side of the 2D-bucket gather: k_put_to_owner across nodes, CE +
// k_store_release within a node. group = GROUP_DENSE / GROUP_MOE; owner is a
// group rank of that group; wait_event = the local event g_stream_gather_put
// waits on first
int gather_put_to_owner(int group,
                        int slot,
                        size_t send_offset,
                        int64_t shard_numel,
                        int owner,
                        size_t arrive_offset,
                        size_t intra_arrive_offset,
                        uint64_t seq,
                        int* num_signals,
                        cudaEvent_t wait_event);

// ----------------------------------------------------------------
// Release layer (release.cu): decides whether one stage of one bucket can be
// submitted to the GPU, and submits it. The main thread (advance) and the tail
// thread share the same decision code and the same state.
// Stage ids. Python reads them by name (constants); the values must not change
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
};  // 2D-bucket gather put (Muon)
// Stage sets for advance, OR-ed together. Window A / B (DeepEP has just
// finished, the NIC is idle) uses intra-node copy + put; the pre window (the
// compute stream waits for DeepEP, the SMs are idle) uses intra-node reduce +
// shard reduce.
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
// Constant table exported to Python (name -> value): ADVANCE_* / STAGE_* /
// COUNTER_* / TAIL_*. The values are defined only in release.cu; Python must
// read them by name and never hard-code them
int constants(std::vector<std::string>& names, std::vector<int>& values);
int setup_gather_table(const std::vector<long long>& table);
int gather_put_entry(int entry_index, unsigned long long seq, int* num_signals);
int set_gather_sequences(const std::vector<unsigned long long>& seqs);
int get_gather_sent(std::vector<long long>& flat);
}  // namespace release

// ----------------------------------------------------------------
// Gets the ncclComm_t that Paddle's ProcessGroup already holds for a ring id
// (the .id of the Python group object), together with the group size and this
// rank's rank (implemented in paddle_comm.cc). On a non-zero return, *out_err
// holds the reason.
int paddle_nccl_comm(int ring_id,
                     int device_id,
                     ncclComm_t* out_comm,
                     int* out_size,
                     int* out_rank,
                     std::string* out_err);

// ----------------------------------------------------------------
// GradCommEngine: exported as paddle.base.core.gin_overlap.GradCommEngine
// (registered in paddle/fluid/pybind/gin_overlap_api.cc); one object per
// process. Python calls these methods by name, so method names and py::arg
// names must not change.
// Conventions:
//   - Methods return an int code: 0 = success, < 10000 = cudaError_t,
//     >= 10000 = 10000 + ncclResult_t. Extra results go through out-params,
//     which gin_overlap_api.cc packs into a tuple.
//   - Pointers are passed as uintptr_t; group = GROUP_DENSE(0) / GROUP_MOE(1);
//     an offset is a byte offset in the matching window; numel / shard_numel
//     count elements.
//   - A few methods return something else (a state, a version); their
//     declarations say so.
//   - Streams are never passed from Python. Methods that run on the compute
//     stream get it through paddle_compute_stream; gather_gate /
//     gather_wait_bucket / gather_mark_ready run on g_stream_gather_gate.
//   - send_window_tensor / copy_2d (implemented in paddle_tensor.cc) throw on
//     error instead of returning a code.
// Call order:
//   - init: set_comms -> alloc_windows -> alloc_host_counters ->
//     create_dev_comm -> set_group_topology -> configure -> (Muon)
//     gather_setup -> release_setup.
//   - each backward pass: record_window_mark -> advance (release inside a
//     window) -> tail_mark / tail_start / bucket_ready / tail_all_posted /
//     tail_wait / tail_finish (tail thread) -> join_bucket_streams ->
//     end_step -> release_step_reset.
//   - release progress: release_status / release_stuck_report.
// The expert stages (submit_moe_put / submit_moe_reduce) have no intra_node /
// inter_node in their names: expert buckets have no intra-node stage, and
// expert peers may sit on the same node.
class GradCommEngine {
 public:
  GradCommEngine() = default;

  static const char* error_string(int code);
  // NCCL_VERSION_CODE of the loaded libnccl / of the headers used at build time
  static int nccl_runtime_version();
  static int nccl_build_version();

  // Takes Paddle's existing comms. A ring id < 0 means the group is unused
  // (no expert buckets). Returns an error, with the reason in err, when a comm
  // cannot be found or its size / rank differs from the values passed in
  int set_comms(int dense_ring_id,
                int expert_ring_id,
                int device_id,
                int dense_size,
                int dense_rank,
                int expert_size,
                int expert_rank,
                std::string* err);
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
  // The only place bucket counts are passed in: creates the side streams,
  // per-bucket events and the window mark ring, and takes the byte offsets of
  // the three credit areas in the host windows (credit_intra / credit_inter in
  // the dense group, credit_moe in the expert group)
  int configure(int num_dense_buckets,
                int num_moe_buckets,
                int num_window_marks,
                size_t credit_intra_offset,
                size_t credit_inter_offset,
                size_t credit_moe_offset);
  // Records a window mark on the compute stream and writes its slot to *slot.
  // cudaErrorNotReady without configure.
  int record_window_mark(int* slot);

  int alloc_host_counters(int group,
                          size_t want_bytes,
                          uintptr_t* cpu_ptr,
                          size_t* nbytes);
  // kind: 0 = dense bucket, 1 = expert bucket (as in bucket_ready)
  int join_bucket_streams(int kind, int bucket);
  int scale_range(int group, size_t send_offset, int64_t numel, float scale);
  int end_step();
  void stats(uint64_t* out3);

  // ---- Python entry points of the release layer (release.cu)
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
  // bucket_ready / tail_all_posted return 0 = accepted, 1 = the tail thread
  // stopped on an error (see release_error), 2 = wrong usage
  int bucket_ready(int kind, int bucket, float scale, int mark);
  int release_status(std::vector<int>& positions,
                     std::vector<int>& counts,
                     std::vector<int>& posted,
                     int* done);
  int release_stuck_report(std::vector<long long>& flat);
  int release_error(int* code, int* stage, int* bucket);
  // What the release layer is submitting right now: stage *
  // ACTIVITY_STAGE_SCALE + bucket + 1; 0 = not inside a CUDA call
  int release_activity();
  // Returns the tail mark slot (>= 0); -1 = no release_setup or
  // cudaEventRecord failed, -2 = this step's tail mark slots are used up
  int tail_mark();
  int tail_start(int intra_node_reduce_blocks,
                 int shard_reduce_blocks,
                 int mark,
                 int all_posted);
  int tail_all_posted();
  // tail_wait / tail_abort return the tail thread's state (TAIL_*; TAIL_RUN =
  // still running after the timeout), or -1 without release_setup. They block
  // for up to timeout_s / wait_s seconds.
  int tail_wait(double timeout_s);
  int tail_abort(double wait_s);
  int tail_finish();
  int tail_gather_setup(const std::vector<long long>& table);
  int tail_gather_set_sequences(const std::vector<unsigned long long>& seqs);
  int tail_gather_get_sent(std::vector<long long>& flat);

  int gather_setup(int num_slots);
  // The next three run on the gate stream of the 2D gather
  // (g_stream_gather_gate)
  int gather_gate(int mark);
  int gather_wait_bucket(int kind, int bucket);
  int gather_mark_ready(int slot);
  // Main thread sends the gather put of one 2D bucket: only the table index
  // and this step's sequence number are passed; slot / offsets / owner come
  // from the table built by release::setup_gather_table
  int gather_put_entry(int entry_index,
                       unsigned long long seq,
                       int* num_signals);
  int gather_join(int slot);
};
}  // namespace gin_overlap
