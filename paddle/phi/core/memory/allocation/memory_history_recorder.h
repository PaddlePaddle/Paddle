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

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "paddle/common/macros.h"
#include "paddle/phi/core/memory/allocation/spin_lock.h"

namespace paddle {
namespace memory {

// Actions recorded in the unified GPU memory history timeline.
enum class MemHistoryAction {
  kAlloc,
  kFreeRequested,
  kFreeCompleted,
  kSegmentAlloc,
  kSegmentFree,
  kSegmentMap,
  kSegmentUnmap,
  kOom,
  kSnapshot,
  kAnnotation,
};

// A single event in the per-device memory history ring buffer.
struct MemHistoryTraceEntry {
  MemHistoryAction action;
  int device;
  uintptr_t addr;
  size_t size;
  uint64_t id;
  uint64_t stream;
  uint64_t time_us;
  std::string op_name;  // Innermost op label (see g_mem_op_label_stack).
  // Opaque id of the captured Python dispatch stack (interned in the pybind
  // layer, resolved to frames at snapshot time). 0 == no stack captured
  // (recording disabled, size below threshold, or a non-dispatch/backward
  // allocation). Only populated for kAlloc events.
  uint64_t stack_id = 0;
};

// Global on/off switch checked inline at every hook site so that recording is
// zero-overhead when disabled. Exported: the inline accessors below are
// instantiated in the pybind / generated-eager targets, so the definition must
// be visible across the phi DLL boundary on Windows.
PADDLE_API extern std::atomic<bool> g_mem_history_enabled;

inline bool MemHistoryEnabled() {
  return g_mem_history_enabled.load(std::memory_order_relaxed);
}

// Record a memory history event into the recorder for `device`. Cheap no-op
// when recording is disabled (guarded internally as well). Out-of-line so that
// the allocator hot path only inlines the enabled check.
void PADDLE_API RecordMemHistory(MemHistoryAction action,
                                 int device,
                                 uintptr_t addr,
                                 size_t size,
                                 uint64_t id,
                                 uint64_t stream);

// Per-thread stack of the op name currently executing on this thread. In eager
// mode the op dispatch -> kernel -> Alloc chain is synchronous on one thread,
// so the innermost (back) label is the op that triggered an allocation.
// Entries are static string literals (or member strings alive for the guard's
// lifetime); we store raw `const char*` to avoid per-op heap traffic.
// Exported because MemLabelGuard is inlined into the generated eager code,
// which lives outside the phi DLL.
PADDLE_API extern thread_local std::vector<const char*> g_mem_op_label_stack;

// Returns the innermost op label active on this thread, or nullptr if none.
inline const char* CurrentMemOpLabel() {
  return g_mem_op_label_stack.empty() ? nullptr : g_mem_op_label_stack.back();
}

// Per-thread stack of the opaque Python-dispatch-stack id captured at op /
// PyLayer entry (in the pybind layer, while the GIL is still held, before
// PyEval_SaveThread). The allocator hot path only reads the innermost id --
// it never touches Python, so recording stays lock/GIL-free and deadlock-free.
//
// This is a STACK (not a flat slot) mirroring g_mem_op_label_stack: an outer
// entry point (e.g. PyLayer.apply, a legacy collective op) pushes the call-site
// stack, and nested eager ops push their own finer stacks on top and pop back
// on exit. So an allocation made directly by an outer wrapper (a comm buffer,
// a raw C++ alloc between nested ops) still inherits the wrapper's stack
// instead of falling back to 0. 0 == no stack captured.
// Exported because MemStackGuard is inlined into the pybind / generated eager
// targets, which live outside the phi DLL.
PADDLE_API extern thread_local std::vector<uint64_t> g_mem_stack_id_stack;

// Returns the innermost captured stack id active on this thread, or 0 if none.
inline uint64_t CurrentMemStackId() {
  return g_mem_stack_id_stack.empty() ? 0 : g_mem_stack_id_stack.back();
}

// RAII guard pushed at op-dispatch / PyLayer entry (while the GIL is held).
// Pushes only when recording is enabled (captured at construction so push/pop
// stay balanced even if recording is toggled mid-op); zero-overhead (one
// relaxed atomic load) when disabled. Mirrors MemLabelGuard.
class MemStackGuard {
 public:
  explicit MemStackGuard(uint64_t stack_id) : active_(MemHistoryEnabled()) {
    if (active_) g_mem_stack_id_stack.push_back(stack_id);
  }
  ~MemStackGuard() {
    if (active_ && !g_mem_stack_id_stack.empty()) {
      g_mem_stack_id_stack.pop_back();
    }
  }
  MemStackGuard(const MemStackGuard&) = delete;
  MemStackGuard& operator=(const MemStackGuard&) = delete;

 private:
  bool active_;
};

// Minimum allocation size (bytes) for which a Python stack is attributed. Kept
// as an atomic so it can be tuned per recording session without touching the
// hot path's cost when recording is disabled. Exported: SetMemStackMinSize is
// inlined into pybind.cc, outside the phi DLL.
PADDLE_API extern std::atomic<size_t> g_mem_stack_min_size;

inline void SetMemStackMinSize(size_t min_size) {
  g_mem_stack_min_size.store(min_size, std::memory_order_relaxed);
}

// RAII guard pushed at op-dispatch entry (forward and backward). Pushes only
// when recording is enabled (captured at construction so push/pop stay
// balanced even if recording is toggled mid-op); zero-overhead (one relaxed
// atomic load) when disabled.
class MemLabelGuard {
 public:
  explicit MemLabelGuard(const char* name) : active_(MemHistoryEnabled()) {
    if (active_) g_mem_op_label_stack.push_back(name);
  }
  ~MemLabelGuard() {
    if (active_ && !g_mem_op_label_stack.empty()) {
      g_mem_op_label_stack.pop_back();
    }
  }
  MemLabelGuard(const MemLabelGuard&) = delete;
  MemLabelGuard& operator=(const MemLabelGuard&) = delete;

 private:
  bool active_;
};

// Per-device fixed-capacity ring buffer of memory history events. Modeled on
// torch's RingBuffer: once full, new entries overwrite the oldest in a circular
// fashion. GetTrace() returns entries in chronological (oldest-first) order.
// Exported: Instance/SetEnabled/Annotate/GetTrace are called from pybind.cc,
// outside the phi DLL.
class PADDLE_API MemoryHistoryRecorder {
 public:
  static MemoryHistoryRecorder& Instance();

  // Enable/disable recording. When enabling, (re)sets the per-device ring
  // capacity to `max_entries` and clears any previously recorded events.
  void SetEnabled(bool enabled, size_t max_entries);

  // Append an event to the ring for `device`. No-op if recording is disabled.
  void Record(const MemHistoryTraceEntry& entry);

  // Insert a user annotation (a named time-marker, e.g. "backward begin") into
  // every device ring that currently exists, stamped with the current time.
  // No-op if recording is disabled. Used to mark
  // forward/backward/step/recompute boundaries on the memory timeline.
  void Annotate(const std::string& msg);

  // Return the events for `device` in chronological order. Empty if `device`
  // is out of range or nothing has been recorded.
  std::vector<MemHistoryTraceEntry> GetTrace(int device);

  // Drop all recorded events across all devices (keeps capacity/enabled state).
  void Clear();

 private:
  MemoryHistoryRecorder() = default;

  struct DeviceRing {
    std::vector<MemHistoryTraceEntry> buf;
    size_t next = 0;
    bool full = false;
    SpinLock lock;
  };

  // Grow rings_ so that index `device` is valid. Caller must hold rings_lock_.
  DeviceRing& EnsureDevice(int device);

  std::vector<std::unique_ptr<DeviceRing>> rings_;
  SpinLock rings_lock_;
  // Ring capacity. Written by SetEnabled() under rings_lock_ but read by
  // Record() on allocating threads without it, so it must be atomic: a
  // reconfigure/disable from Python can run concurrently with allocations.
  // Relaxed is sufficient -- buf/next are always mutated under the per-ring
  // lock, so this value only needs to be untorn, not ordered against them.
  std::atomic<size_t> capacity_{0};
};

}  // namespace memory
}  // namespace paddle
