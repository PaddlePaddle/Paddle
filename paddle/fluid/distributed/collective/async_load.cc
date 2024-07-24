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

namespace paddle {
namespace distributed {

AsyncLoad::Task::Task(const Place& place)
    : load_event_(place, platform::GenerateDeviceEventFlag()),
      task_place_(place) {}

AsyncLoad::Task::~Task() {}

bool AsyncLoad::Task::IsCompleted() { return load_event_.Query(); }

void AsyncLoad::Task::Synchronize() {
  const auto* calc_ctx = phi::DeviceContextPool::Instance().Get(task_place_);
  load_event_.Wait(platform::Place2DeviceType(task_place_), calc_ctx);
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
      phi::errors::InvalidArgument(
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
      phi::errors::InvalidArgument(
          "AsyncLoad::Reload only support GPUPinned -> GPU now."));

  PADDLE_ENFORCE_EQ(is_initialized_,
                    true,
                    phi::errors::PreconditionNotMet(
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
