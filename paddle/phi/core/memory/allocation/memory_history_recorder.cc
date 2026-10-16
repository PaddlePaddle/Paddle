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

#include "paddle/phi/core/memory/allocation/memory_history_recorder.h"

#include <mutex>

#include "paddle/phi/core/os_info.h"

namespace paddle {
namespace memory {

namespace {
// All state stays internal to this TU: exported data does not work here (see
// MemHistoryEnabled in the header), and thread_local data cannot carry a dll
// interface on MSVC (C2492). Consumers go through the exported functions.
std::atomic<bool> g_mem_history_enabled{false};
std::atomic<size_t> g_mem_stack_min_size{0};
thread_local std::vector<const char*> g_mem_op_label_stack;
thread_local std::vector<uint64_t> g_mem_stack_id_stack;
}  // namespace

bool MemHistoryEnabled() {
  return g_mem_history_enabled.load(std::memory_order_relaxed);
}

void SetMemStackMinSize(size_t min_size) {
  g_mem_stack_min_size.store(min_size, std::memory_order_relaxed);
}

std::vector<const char*>& MemOpLabelStack() { return g_mem_op_label_stack; }

std::vector<uint64_t>& MemStackIdStack() { return g_mem_stack_id_stack; }

MemoryHistoryRecorder& MemoryHistoryRecorder::Instance() {
  static MemoryHistoryRecorder instance;
  return instance;
}

MemoryHistoryRecorder::DeviceRing& MemoryHistoryRecorder::EnsureDevice(
    int device) {
  if (static_cast<size_t>(device) >= rings_.size()) {
    rings_.resize(device + 1);
  }
  if (rings_[device] == nullptr) {
    rings_[device] = std::make_unique<DeviceRing>();
  }
  return *rings_[device];
}

void MemoryHistoryRecorder::SetEnabled(bool enabled, size_t max_entries) {
  if (!enabled) {
    // Publish the disable BEFORE clearing. The ring lock's release/acquire then
    // guarantees that a writer reaching its critical section after the clear
    // observes it and drops its event instead of resurrecting a cleared ring.
    g_mem_history_enabled.store(false, std::memory_order_relaxed);
  }
  {
    std::lock_guard<SpinLock> lock(rings_lock_);
    // Disabling forces the capacity to 0 whatever the caller passed (Python
    // sends its default max_entries even when disabling), so both the fast
    // reject and the in-lock check in Record() bail out afterwards.
    capacity_.store(enabled ? max_entries : 0, std::memory_order_relaxed);
    for (auto& ring : rings_) {
      if (ring == nullptr) continue;
      std::lock_guard<SpinLock> ring_lock(ring->lock);
      ring->buf.clear();
      ring->next = 0;
      ring->full = false;
    }
  }
  if (enabled) {
    // Publish the enable last, so no writer enters with a stale capacity.
    g_mem_history_enabled.store(true, std::memory_order_relaxed);
  }
}

void MemoryHistoryRecorder::Record(const MemHistoryTraceEntry& entry) {
  if (entry.device < 0) return;
  // Fast reject before touching any lock.
  if (capacity_.load(std::memory_order_relaxed) == 0) return;

  DeviceRing* ring = nullptr;
  {
    std::lock_guard<SpinLock> lock(rings_lock_);
    ring = &EnsureDevice(entry.device);
  }

  std::lock_guard<SpinLock> ring_lock(ring->lock);
  // Re-check the enabled state here: SetEnabled() publishes a disable before
  // clearing the rings under this same lock, so anything arriving after the
  // clear must drop its event -- otherwise a straggler that passed the check in
  // RecordMemHistory() could make events reappear after recording was stopped.
  if (!MemHistoryEnabled()) return;
  // Re-read the capacity too: keeps the size check and the modulo consistent,
  // honours a capacity lowered while we took locks, and the zero check must be
  // repeated or `% cap` could divide by zero.
  const size_t cap = capacity_.load(std::memory_order_relaxed);
  if (cap == 0) return;
  if (ring->buf.size() < cap) {
    // Not yet at capacity: append.
    ring->buf.push_back(entry);
  } else {
    // At capacity: circular overwrite of the oldest entry.
    ring->buf[ring->next] = entry;
    ring->next = (ring->next + 1) % cap;
    ring->full = true;
  }
}

void MemoryHistoryRecorder::Annotate(const std::string& msg) {
  if (!MemHistoryEnabled()) return;
  // Snapshot the set of existing device rings first; don't hold rings_lock_
  // across Record() (which re-acquires it and would deadlock the SpinLock).
  std::vector<int> devices;
  {
    std::lock_guard<SpinLock> lock(rings_lock_);
    for (size_t i = 0; i < rings_.size(); ++i) {
      if (rings_[i] != nullptr) devices.push_back(static_cast<int>(i));
    }
  }
  uint64_t now = phi::PosixInNsec() / 1000;
  for (int dev : devices) {
    MemHistoryTraceEntry entry;
    entry.action = MemHistoryAction::kAnnotation;
    entry.device = dev;
    entry.addr = 0;
    entry.size = 0;
    entry.id = 0;
    entry.stream = 0;
    entry.time_us = now;
    entry.op_name = msg;
    Record(entry);
  }
}

std::vector<MemHistoryTraceEntry> MemoryHistoryRecorder::GetTrace(int device) {
  if (device < 0) return {};
  DeviceRing* ring = nullptr;
  {
    std::lock_guard<SpinLock> lock(rings_lock_);
    if (static_cast<size_t>(device) >= rings_.size() ||
        rings_[device] == nullptr) {
      return {};
    }
    ring = rings_[device].get();
  }

  std::lock_guard<SpinLock> ring_lock(ring->lock);
  std::vector<MemHistoryTraceEntry> out;
  out.reserve(ring->buf.size());
  if (ring->full) {
    // Oldest-first: start at next, wrap around.
    for (size_t i = 0; i < ring->buf.size(); ++i) {
      out.push_back(ring->buf[(ring->next + i) % ring->buf.size()]);
    }
  } else {
    out = ring->buf;
  }
  return out;
}

void MemoryHistoryRecorder::Clear() {
  std::lock_guard<SpinLock> lock(rings_lock_);
  for (auto& ring : rings_) {
    if (ring == nullptr) continue;
    std::lock_guard<SpinLock> ring_lock(ring->lock);
    ring->buf.clear();
    ring->next = 0;
    ring->full = false;
  }
}

void RecordMemHistory(MemHistoryAction action,
                      int device,
                      uintptr_t addr,
                      size_t size,
                      uint64_t id,
                      uint64_t stream) {
  if (!MemHistoryEnabled()) return;
  MemHistoryTraceEntry entry;
  entry.action = action;
  entry.device = device;
  entry.addr = addr;
  entry.size = size;
  entry.id = id;
  entry.stream = stream;
  entry.time_us = phi::PosixInNsec() / 1000;
  const char* label = CurrentMemOpLabel();
  if (label != nullptr) entry.op_name = label;
  if (action == MemHistoryAction::kAlloc &&
      size >= g_mem_stack_min_size.load(std::memory_order_relaxed)) {
    // Pure thread_local read: no GIL, no lock. The id was stamped at op /
    // PyLayer dispatch (pybind layer) while the GIL was still held.
    entry.stack_id = CurrentMemStackId();
  }
  MemoryHistoryRecorder::Instance().Record(entry);
}

}  // namespace memory
}  // namespace paddle
