// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
  const auto* calc_ctx =
      platform::DeviceContextPool::Instance().Get(task_place_);
  load_event_.Wait(platform::Place2DeviceType(task_place_), calc_ctx);
}

void AsyncLoad::Task::UpdateWaitChain(const phi::DeviceContext& ctx) {
  load_event_.Record(&ctx);
}

std::shared_ptr<AsyncLoad::Task> AsyncLoad::CreateTask(const Place& place) {
  return std::make_shared<AsyncLoad::Task>(place);
}

void AsyncLoad::PrepareLoadEnv(const std::string& key) {
  if (place_to_calc_event_.find(key) == place_to_calc_event_.end()) {
    place_to_calc_event_.emplace(
        key, platform::DeviceEvent(place, platform::GenerateDeviceEventFlag()));
    place_to_load_ctx_.emplace(
        key, std::move(std::make_unique<phi::GPUContext>(place)));
  }

  const auto* calc_ctx = static_cast<phi::GPUContext*>(
      platform::DeviceContextPool::Instance().Get(place));
  calc_event.Record(calc_ctx);
  calc_event.Wait(platform::Place2DeviceType(place), async_ctx.get());
}

std::shared_ptr<AsyncLoad::Task> AsyncLoad::Offload(
    phi::DenseTensor* dst, const phi::DenseTensor& src) {
  // GPU -> GPUPinned
  const auto& place = src.place();

  PADDLE_ENFORCE_EQ(
      platform::is_gpu_place(place),
      true,
      platform::errors::InvalidArgument(
          "AsyncLoad::Offload only support GPU -> GPUPinned now."));

  dst->Resize(src.dims());
  auto* dst_ptr = dst->mutable_data(platform::CUDAPinnedPlace(), src.dtype());
  auto* src_ptr = src.data();
  auto size = src.numel() * phi::SizeOf(src.dtype());

  // 1. wait calc stream to finish
  std::string key = "offload";
  PrepareLoadEnv(key);
  auto& async_ctx = place_to_load_ctx_.at(key);

  // 2. copy data from src to dst
  auto stream = async_ctx->stream();
  phi::memory_utils::Copy(
      platform::CUDAPlace(), dst_ptr, place, src_ptr, size, stream);

  // 3. record event on offload stream
  auto task = CreateTask(place);
  task->UpdateWaitChain(*async_ctx);
  return task;
}

std::shared_ptr<AsyncLoad::Task> AsyncLoad::Reload(
    phi::DenseTensor* dst, const phi::DenseTensor& src) {
  // GPUPinned -> GPU
  const auto& place = src.place();
  PADDLE_ENFORCE_EQ(
      platform::is_cuda_pinned_place(place),
      true,
      platform::errors::InvalidArgument(
          "AsyncLoad::Reload only support GPUPinned -> GPU now."));

  dst->Resize(src.dims());
  auto* dst_ptr = dst->mutable_data(platform::CUDAPlace(), src.dtype());
  auto* src_ptr = src.data();
  auto size = src.numel() * phi::SizeOf(src.dtype());

  // 1. wait calc stream to finish
  std::string key = "reload";
  PrepareLoadEnv(key);

  auto& async_ctx = place_to_load_ctx_.at(key);

  // 2. copy data from src to dst
  auto stream = async_ctx->stream();
  phi::memory_utils::Copy(
      platform::CUDAPinnedPlace(), dst_ptr, place, src_ptr, size, stream);

  // 3. record event on offload stream
  auto task = CreateTask(place);
  task->UpdateWaitChain(*async_ctx);
  return task;
}

}  // namespace distributed
}  // namespace paddle
