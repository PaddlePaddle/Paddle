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

#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/framework/offload_vars_pool.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
namespace paddle {

namespace framework {

void ResourceHolder::Alloc(size_t required_size) {
  platform::CUDAPlace place(platform::GetCurrentDeviceId());
  paddle::platform::DeviceContextPool& pool =
      paddle::platform::DeviceContextPool::Instance();
  auto* dev_ctx = static_cast<phi::GPUContext*>(pool.Get(place));
  dev_ctx->Alloc(&tensor_, phi::DataType::FLOAT32, required_size);
}

Buffer::Buffer(size_t required_size) {
  gpu_resource_.Alloc(required_size);
  cudaEventCreate(&event_);
}

Buffer::~Buffer() { cudaEventDestroy(event_); }

OffloadVarsPool::~OffloadVarsPool() { cudaStreamDestroy(load_stream_); }

void OffloadVarsPool::Transfer(const phi::DenseTensor& src,
                               phi::DenseTensor* dst,
                               size_t n,
                               cudaStream_t stream) {
  cudaMemcpyAsync(dst->data(), src.data(), n, cudaMemcpyHostToDevice, stream);
}
void OffloadVarsPool::Init(
    size_t init_size,
    const std::list<LayerIdx2ParamsTensors>& weights_queue,
    const std::unordered_set<size_t>& offload_layers) {
  paddle::platform::DeviceContextPool& pool =
      paddle::platform::DeviceContextPool::Instance();
  platform::Place place = platform::CUDAPlace(platform::GetCurrentDeviceId());
  auto* dev_ctx_ = static_cast<phi::GPUContext*>(pool.Get(place));
  kernel_run_stream_ = dev_ctx_->stream();
  cudaStreamCreateWithFlags(&load_stream_, cudaStreamNonBlocking);

  buffer_nodes_.reserve(2);
  for (size_t i = 0; i < 2u; i++) {
    buffer_nodes_.emplace_back(init_size);
  }

  weights_queue_ = weights_queue;
  weights_queue_cpy_ = weights_queue;
  offload_layers_ = offload_layers;

  for (auto& pair : weights_queue_) {
    CHECK_EQ(pair.second.first.size(), pair.second.second.size());
    for (size_t i = 0; i < pair.second.first.size(); i++) {
      CHECK_EQ(pair.second.first[i]->numel(), pair.second.second[i]->numel());
      CHECK_EQ(pair.second.first[i]->dtype(), pair.second.second[i]->dtype());
    }
  }

  FillPool();
}

void OffloadVarsPool::FillPool() {
  // fill free node
  for (size_t i = 0; i < buffer_nodes_.size(); i++) {
    if (weights_queue_.empty()) return;
    if (buffer_nodes_[i].free_) {
      LayerIdx2ParamsTensors op_params_info = weights_queue_.front();
      weights_queue_.pop_front();
      buffer_nodes_[i].layer_id_ = op_params_info.first;
      cudaStreamWaitEvent(load_stream_, buffer_nodes_[i].event_, 0);
      for (size_t j = 0; j < op_params_info.second.first.size(); j++) {
        auto* src_tensor = op_params_info.second.first[j];
        auto* dst_tensor = op_params_info.second.second[j];
        dst_tensor->set_type(src_tensor->dtype());
        dst_tensor->Resize(src_tensor->dims());
        buffer_nodes_[i].gpu_resource_.PartionMemoryTo(
            dst_tensor, dst_tensor->numel() * SizeOf(dst_tensor->dtype()));
        Transfer(*src_tensor,
                 dst_tensor,
                 src_tensor->numel() * SizeOf(src_tensor->dtype()),
                 load_stream_);
      }
      cudaEventRecord(buffer_nodes_[i].event_, load_stream_);
      buffer_nodes_[i].free_ = false;
      active_layers_[buffer_nodes_[i].layer_id_] = i;
    }
  }
}
}  // namespace framework
}  // namespace paddle
#endif
