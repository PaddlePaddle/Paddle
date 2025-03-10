// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/distributed/collective/xpu_async_load.h"

#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/common/memory_utils.h"  // phi::memory_utils::Copy
// #include "paddle/phi/core/device_context_pool.h" // for DeviceContextPool
// #include "paddle/phi/core/places.h"              // phi::is_xpu_place(...)
#include "paddle/phi/core/compat/convert_utils.h"

namespace paddle {
namespace distributed {

using phi::is_xpu_place;

/**
 * Helper: Insert or retrieve a DeviceEvent in the map without
 * default-constructing it.
 *   - If place is XPU, we skip event usage entirely (dummy).
 *   - If place is NOT XPU, we create a DeviceEvent with the needed constructor.
 */
static platform::DeviceEvent& GetOrCreateEvent(
    std::unordered_map<std::string, platform::DeviceEvent>* event_map,
    const std::string& key,
    const phi::Place& place) {
  // If it's XPU, we do a "dummy" CPU-based event or skip
  // (but let's store a CPU event just so we can return a reference).
  // In a real design, you might do a separate approach.

  phi::Place event_place = is_xpu_place(place) ? phi::CPUPlace() : place;
  unsigned int flags = platform::GenerateDeviceEventFlag();

  auto it = event_map->find(key);
  if (it == event_map->end()) {
    // Insert using piecewise_construct to avoid default constructor
    auto emplace_result =
        event_map->emplace(std::piecewise_construct,
                           std::forward_as_tuple(key),
                           std::forward_as_tuple(event_place, flags));
    it = emplace_result.first;  // newly inserted
  }
  return it->second;
}

/* ------------------- Task Implementation ------------------- */

XpuAsyncLoad::Task::Task(const Place& place)
    : use_event_(!is_xpu_place(place)),
      // If place is XPU, we store a CPU event just so load_event_ is valid
      // (some dummy fallback, we won't really use it)
      load_event_(use_event_ ? place : phi::CPUPlace(),
                  platform::GenerateDeviceEventFlag()),
      task_place_(place) {}

XpuAsyncLoad::Task::~Task() = default;

bool XpuAsyncLoad::Task::IsCompleted() {
  if (!use_event_) {
    // For XPU, skip real event usage and just say "complete"
    return true;
  }
  return load_event_.Query();
}

// Example fix in Task::XpuSynchronize():
void XpuAsyncLoad::Task::XpuSynchronize() {
  if (!use_event_) {
    return;
  }
  auto* calc_ctx = phi::DeviceContextPool::Instance().Get(task_place_);
  // OLD (won't compile in your version):
  //   auto backend = task_place_.GetBackend();
  //   load_event_.Wait(backend, calc_ctx);
  // NEW:
  load_event_.Wait(platform::Place2DeviceType(task_place_), calc_ctx);
}

void XpuAsyncLoad::Task::CpuSynchronize() {
  if (!use_event_) {
    return;
  }
  load_event_.Finish();
}

void XpuAsyncLoad::Task::UpdateWaitChain(const phi::DeviceContext& ctx) {
  if (!use_event_) {
    // skip
    return;
  }
  load_event_.Record(&ctx);
}

/* ------------------- XpuAsyncLoad Implementation ------------------- */

std::shared_ptr<XpuAsyncLoad::Task> XpuAsyncLoad::CreateTask(
    const Place& place) {
  return std::make_shared<XpuAsyncLoad::Task>(place);
}

void XpuAsyncLoad::PrepareLoadEnv(const std::string& key, const Place& place) {
  if (!is_initialized_) {
    is_initialized_ = true;
    xpu_place_ = place;
    // If not XPU, create a real event; if XPU, we store a dummy CPU event
    (void)GetOrCreateEvent(&place_to_calc_event_, key, place);

    // Create an XPUContext for the offload
    load_ctx_ = std::make_unique<phi::XPUContext>(place);
  }
}

// Another fix in SyncCalcuStream():
void XpuAsyncLoad::SyncCalcuStream(const Place& place,
                                   phi::XPUContext* offload_ctx,
                                   platform::DeviceEvent* calc_event) {
  if (is_xpu_place(place)) {
    // skip or do fallback
    return;
  }
  auto* calc_ctx = phi::DeviceContextPool::Instance().Get(place);
  calc_event->Record(calc_ctx);
  // OLD (won't compile):
  //   auto backend = place.GetBackend();
  //   calc_event.Wait(backend, offload_ctx);
  // NEW:
  calc_event->Wait(platform::Place2DeviceType(place), offload_ctx);
}

/* ------------ Offload (XPU -> CPU pinned or CPU) ------------ */
std::shared_ptr<XpuAsyncLoad::Task> XpuAsyncLoad::Offload(
    phi::DenseTensor* dst, const phi::DenseTensor& src) {
  PADDLE_ENFORCE_EQ(
      is_xpu_place(src.place()),
      true,
      phi::errors::InvalidArgument("Offload only supports XPU source."));

  std::string key = "load_key";
  PrepareLoadEnv(key, src.place());
  // retrieve or create the event
  auto& calc_event = GetOrCreateEvent(&place_to_calc_event_, key, src.place());
  // sync
  SyncCalcuStream(xpu_place_, load_ctx_.get(), &calc_event);

  // do synchronous copy to CPU
  dst->Resize(src.dims());
  size_t size = src.numel() * phi::SizeOf(src.dtype());
  auto cpu_place = phi::CPUPlace();
  auto* cpu_ctx = phi::DeviceContextPool::Instance().Get(cpu_place);
  void* dst_ptr = cpu_ctx->Alloc(dst, src.dtype(), size);
  const void* src_ptr = src.data();

  phi::memory_utils::Copy(cpu_place,
                          dst_ptr,
                          src.place(),
                          src_ptr,
                          size,
                          /*stream=*/nullptr);

  auto task = CreateTask(src.place());
  task->UpdateWaitChain(*load_ctx_);
  return task;
}

/* ------------ OffloadWithOffset (XPU -> CPU partial) ------------ */
std::shared_ptr<XpuAsyncLoad::Task> XpuAsyncLoad::OffloadWithOffset(
    phi::DenseTensor* dst,
    const phi::DenseTensor& src,
    size_t dst_offset,
    size_t src_offset,
    size_t offload_size) {
  PADDLE_ENFORCE_EQ(
      is_xpu_place(src.place()),
      true,
      phi::errors::InvalidArgument("OffloadWithOffset requires XPU source."));

  PADDLE_ENFORCE_EQ(dst->initialized(),
                    true,
                    phi::errors::PreconditionNotMet(
                        "dst must be initialized for partial offload."));

  PADDLE_ENFORCE_LE(
      src_offset + offload_size,
      src.numel(),
      phi::errors::InvalidArgument("src offset + size out of range."));
  PADDLE_ENFORCE_LE(
      dst_offset + offload_size,
      dst->numel(),
      phi::errors::InvalidArgument("dst offset + size out of range."));

  std::string key = "load_key";
  PrepareLoadEnv(key, src.place());
  auto& calc_event = GetOrCreateEvent(&place_to_calc_event_, key, src.place());
  SyncCalcuStream(xpu_place_, load_ctx_.get(), &calc_event);

  size_t elem_size = phi::SizeOf(src.dtype());
  size_t copy_bytes = offload_size * elem_size;
  const void* src_ptr =
      static_cast<const char*>(src.data()) + src_offset * elem_size;
  void* dst_ptr = static_cast<char*>(dst->data()) + dst_offset * elem_size;

  phi::memory_utils::Copy(dst->place(),
                          dst_ptr,
                          src.place(),
                          src_ptr,
                          copy_bytes,
                          /*stream=*/nullptr);

  auto task = CreateTask(src.place());
  task->UpdateWaitChain(*load_ctx_);
  return task;
}

/* ------------ Reload (CPU -> XPU) ------------ */
std::shared_ptr<XpuAsyncLoad::Task> XpuAsyncLoad::Reload(
    phi::DenseTensor* dst, const phi::DenseTensor& src) {
  PADDLE_ENFORCE_EQ(
      is_initialized_,
      true,
      phi::errors::PreconditionNotMet("Call Offload before Reload."));

  // Possibly we check if src is CPU or pinned place
  // We'll skip that check or treat it as CPU place
  std::string key = "load_key";
  auto& calc_event = GetOrCreateEvent(&place_to_calc_event_, key, xpu_place_);
  SyncCalcuStream(xpu_place_, load_ctx_.get(), &calc_event);

  // Now do CPU->XPU
  dst->Resize(src.dims());
  size_t size = src.numel() * phi::SizeOf(src.dtype());

  auto* xpu_ctx = phi::DeviceContextPool::Instance().Get(xpu_place_);
  void* dst_ptr = xpu_ctx->Alloc(dst, src.dtype(), size, /*pinned=*/false);
  const void* src_ptr = src.data();

  phi::memory_utils::Copy(xpu_place_,
                          dst_ptr,
                          src.place(),
                          src_ptr,
                          size,
                          /*stream=*/nullptr);

  auto task = CreateTask(xpu_place_);
  task->UpdateWaitChain(*load_ctx_);
  return task;
}

}  // namespace distributed
}  // namespace paddle
