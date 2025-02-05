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

#include "paddle/fluid/distributed/collective/async_load.h"
#include "paddle/phi/common/memory_utils.h"

COMMON_DECLARE_bool(use_stream_safe_cuda_allocator);
COMMON_DECLARE_bool(use_cuda_malloc_async_allocator);

namespace paddle {
namespace distributed {

AsyncLoad::Task::Task(const Place& place)
    : load_event_(place, platform::GenerateDeviceEventFlag()),
      task_place_(place) {}

AsyncLoad::Task::~Task() {}

bool AsyncLoad::Task::IsCompleted() { return load_event_.Query(); }

void AsyncLoad::Task::CudaSynchronize() {
  const auto* calc_ctx =
      platform::DeviceContextPool::Instance().Get(task_place_);
  load_event_.Wait(platform::Place2DeviceType(task_place_), calc_ctx);
}

void AsyncLoad::Task::CpuSynchronize() {
  // cudaEventSynchronize
  load_event_.Finish();
}

void AsyncLoad::Task::UpdateWaitChain(const phi::DeviceContext& ctx) {
  load_event_.Record(&ctx);
}

std::shared_ptr<AsyncLoad::Task> AsyncLoad::CreateTask(const Place& place) {
  return std::make_shared<AsyncLoad::Task>(place);
}

void AsyncLoad::SyncCalcuStream(const Place& place,
                                phi::GPUContext* ctx,
                                platform::DeviceEvent& calc_event) {  // NOLINT
  const auto* calc_ctx = static_cast<phi::GPUContext*>(
      phi::DeviceContextPool::Instance().Get(place));
  calc_event.Record(calc_ctx);
  calc_event.Wait(platform::Place2DeviceType(place), ctx);
}

std::shared_ptr<AsyncLoad::Task> AsyncLoad::Offload(
    phi::DenseTensor* dst, const phi::DenseTensor& src) {
  // GPU -> GPUPinned
  const auto& place = src.place();

  PADDLE_ENFORCE_EQ(
      phi::is_gpu_place(place),
      true,
      common::errors::InvalidArgument(
          "AsyncLoad::Offload only support GPU -> GPUPinned now."));

  dst->Resize(src.dims());
  auto size = src.numel() * phi::SizeOf(src.dtype());
  auto* dev_ctx = static_cast<phi::GPUContext*>(
      phi::DeviceContextPool::Instance().Get(place));
  auto* dst_ptr = dev_ctx->Alloc(dst, src.dtype(), size, true);
  auto* src_ptr = src.data();

  // 1. wait calc stream to finish
  std::string key = "load";

  if (!is_initialized_) {
    is_initialized_ = true;
    gpu_place_ = place;
    place_to_calc_event_.emplace(
        key, platform::DeviceEvent(place, platform::GenerateDeviceEventFlag()));
    load_ctx_ = std::make_unique<phi::GPUContext>(place);
  }
  SyncCalcuStream(gpu_place_, load_ctx_.get(), place_to_calc_event_.at(key));

  // 2. copy data from src to dst
  auto stream = load_ctx_->stream();
  phi::memory_utils::Copy(
      dst->place(), dst_ptr, src.place(), src_ptr, size, stream);

  if (FLAGS_use_stream_safe_cuda_allocator ||
      FLAGS_use_cuda_malloc_async_allocator) {
    memory::RecordStream(src.Holder(), stream);
  }

  // 3. record event on offload stream
  auto task = CreateTask(place);
  task->UpdateWaitChain(*load_ctx_);
  return task;
}

std::shared_ptr<AsyncLoad::Task> AsyncLoad::OffloadWithOffset(
    phi::DenseTensor* dst,
    const phi::DenseTensor& src,
    size_t dst_offset,
    size_t src_offset,
    size_t offload_size) {
  // GPU -> GPUPinned
  const auto& place = src.place();

  PADDLE_ENFORCE_EQ(
      phi::is_gpu_place(place),
      true,
      common::errors::InvalidArgument(
          "AsyncLoad::OffloadWithOffset only support GPU src now."));

  PADDLE_ENFORCE_EQ(dst->initialized(),
                    true,
                    common::errors::PreconditionNotMet(
                        "AsyncLoad::OffloadWithOffset only support on "
                        "initialized tensors for both dst and src now."));

  PADDLE_ENFORCE_LE(
      src_offset + offload_size,
      src.numel(),
      common::errors::InvalidArgument(
          "AsyncLoad::OffloadWithOffset src_offset + offload_size should be "
          "less than or equal to src tensor size."));

  PADDLE_ENFORCE_LE(
      dst_offset + offload_size,
      dst->numel(),
      common::errors::InvalidArgument(
          "AsyncLoad::OffloadWithOffset dst_offset + offload_size should be "
          "less than or equal to dst tensor size."));

  auto size_in_bytes = offload_size * phi::SizeOf(src.dtype());
  auto src_offset_in_bytes = src_offset * phi::SizeOf(src.dtype());
  auto dst_offset_in_bytes = dst_offset * phi::SizeOf(src.dtype());
  auto* dst_ptr = dst->data();
  auto* src_ptr = src.data();
  auto* dst_ptr_tmp = static_cast<char*>(dst_ptr);
  auto* src_ptr_tmp = static_cast<const char*>(src_ptr);
  dst_ptr = static_cast<void*>(dst_ptr_tmp + dst_offset_in_bytes);
  src_ptr = static_cast<const void*>(src_ptr_tmp + src_offset_in_bytes);

  // 1. wait calc stream to finish
  std::string key = "load";

  if (!is_initialized_) {
    is_initialized_ = true;
    gpu_place_ = place;
    place_to_calc_event_.emplace(
        key, platform::DeviceEvent(place, platform::GenerateDeviceEventFlag()));
    load_ctx_ = std::move(std::make_unique<phi::GPUContext>(place));
  }
  SyncCalcuStream(gpu_place_, load_ctx_.get(), place_to_calc_event_.at(key));

  // 2. copy data from src to dst
  auto stream = load_ctx_->stream();
  phi::memory_utils::Copy(
      dst->place(), dst_ptr, src.place(), src_ptr, size_in_bytes, stream);

  if (FLAGS_use_stream_safe_cuda_allocator ||
      FLAGS_use_cuda_malloc_async_allocator) {
    memory::RecordStream(src.Holder(), stream);
  }

  // 3. record event on offload stream
  auto task = CreateTask(place);
  task->UpdateWaitChain(*load_ctx_);
  return task;
}

std::shared_ptr<AsyncLoad::Task> AsyncLoad::Reload(
    phi::DenseTensor* dst, const phi::DenseTensor& src) {
  // GPUPinned -> GPU
  const auto& place = src.place();
  PADDLE_ENFORCE_EQ(
      phi::is_cuda_pinned_place(place),
      true,
      common::errors::InvalidArgument(
          "AsyncLoad::Reload only support GPUPinned -> GPU now."));

  PADDLE_ENFORCE_EQ(is_initialized_,
                    true,
                    common::errors::PreconditionNotMet(
                        "You should call Offload before Reload."));

  auto* dev_ctx = static_cast<phi::GPUContext*>(
      phi::DeviceContextPool::Instance().Get(gpu_place_));

  dst->Resize(src.dims());
  auto size = src.numel() * phi::SizeOf(src.dtype());
  auto* dst_ptr = dev_ctx->Alloc(dst, src.dtype(), size, false);
  auto* src_ptr = src.data();

  // 1. wait calc stream to finish
  std::string key = "load";

  SyncCalcuStream(gpu_place_, load_ctx_.get(), place_to_calc_event_.at(key));

  // 2. copy data from src to dst
  auto stream = load_ctx_->stream();
  phi::memory_utils::Copy(
      dst->place(), dst_ptr, src.place(), src_ptr, size, stream);

  // 3. record event on offload stream
  auto task = CreateTask(gpu_place_);
  task->UpdateWaitChain(*load_ctx_);
  return task;
}

}  // namespace distributed
}  // namespace paddle
