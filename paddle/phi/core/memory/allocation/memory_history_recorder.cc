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

std::atomic<bool> g_mem_history_enabled{false};

thread_local std::vector<const char*> g_mem_op_label_stack;

thread_local std::vector<uint64_t> g_mem_stack_id_stack;

std::atomic<size_t> g_mem_stack_min_size{0};

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
  {
    std::lock_guard<SpinLock> lock(rings_lock_);
    capacity_.store(max_entries, std::memory_order_relaxed);
    for (auto& ring : rings_) {
      if (ring == nullptr) continue;
      std::lock_guard<SpinLock> ring_lock(ring->lock);
      ring->buf.clear();
      ring->next = 0;
      ring->full = false;
    }
  }
  g_mem_history_enabled.store(enabled, std::memory_order_relaxed);
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
  // Re-read inside the per-ring critical section so that the size check and
  // the modulo below use one consistent value, and so that a capacity change
  // that happened while we were acquiring locks is honoured (otherwise a stale
  // larger capacity could grow this ring past the newly requested bound). The
  // zero check must be repeated: SetEnabled(_, 0) between the fast reject and
  // here would otherwise reach `% cap` with cap == 0.
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
