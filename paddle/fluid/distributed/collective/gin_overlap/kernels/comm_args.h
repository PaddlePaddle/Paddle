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

// Arguments of the GIN primitives of ProcessGroupNCCL, one struct per
// primitive, used by callers and by the kernels of comm.h alike.
//
// A window is a handle returned by ProcessGroupNCCL.register_comm_buffer() or
// ProcessGroupNCCL::AllocHostWindow() on the group; it names the window of
// this rank, and the same handle addresses the matching window on every peer.
// `peer` is a rank of the group. Offsets and sizes are in bytes.
#pragma once

#include <cstddef>
#include <cstdint>

namespace paddle {
namespace distributed {

// ProcessGroupNCCL::Put: a GIN put of `bytes` bytes from `src_offset` of this
// rank's `src_window` to `dst_offset` of `dst_window` on `peer`, followed by
// an increment by 1 of the u64 counter at `counter_offset` of `counter_window`
// on `peer`.
struct PutArgs {
  void* src_window{nullptr};
  size_t src_offset{0};
  void* dst_window{nullptr};
  size_t dst_offset{0};
  size_t bytes{0};
  int peer{0};
  void* counter_window{nullptr};
  size_t counter_offset{0};
};

// ProcessGroupNCCL::IncrementPeerCounter: an increment by 1 of the u64 counter
// at `counter_offset` of `counter_window` on `peer`.
struct IncrementPeerCounterArgs {
  int peer{0};
  void* counter_window{nullptr};
  size_t counter_offset{0};
  // Set by ProcessGroupNCCL, callers leave it empty: the LSA address of the
  // counter when `peer` is in this rank's LSA team (the increment is then a
  // system-scope atomic add), nullptr otherwise (a GIN signal).
  uint64_t* lsa_counter{nullptr};
};

// ProcessGroupNCCL::CopyToPeers: a copy of `bytes` bytes from `src_offset` of
// this rank's `src_window` to `dst_offset` of `dst_window` on `peer`, which
// must be in this rank's LSA team.
struct CopyToPeersArgs {
  void* src_window{nullptr};
  size_t src_offset{0};
  void* dst_window{nullptr};
  size_t dst_offset{0};
  int peer{0};
  size_t bytes{0};
};

}  // namespace distributed
}  // namespace paddle
