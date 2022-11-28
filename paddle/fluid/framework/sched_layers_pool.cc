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

#include "paddle/fluid/framework/sched_layers_pool.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
namespace paddle {

namespace framework {

void ResourceHolder::mutable_data(size_t required_size) {
  platform::CUDAPlace place(platform::GetCurrentDeviceId());
  tensor_.mutable_data(
      place, paddle::experimental::DataType::FLOAT32, required_size);
}

Buffer::Buffer(size_t required_size) {
  gpu_resource.mutable_data(required_size);
  cudaEventCreate(&event);
}

Buffer::~Buffer() {}

SchedLayersPool::~SchedLayersPool() { cudaStreamDestroy(load_stream_); }

void SchedLayersPool::Transfer(const phi::DenseTensor& src,
                               phi::DenseTensor* dst,
                               size_t n,
                               cudaStream_t stream) {
  cudaMemcpyAsync(dst->data(), src.data(), n, cudaMemcpyHostToDevice, stream);
}
void SchedLayersPool::Init(size_t init_size,
                           const std::list<OpIdx2ParamsTensors>& weights_queue,
                           const std::vector<size_t>& sched_layers) {
  paddle::platform::DeviceContextPool& pool =
      paddle::platform::DeviceContextPool::Instance();
  platform::Place place = platform::CUDAPlace(platform::GetCurrentDeviceId());
  auto* dev_ctx_ = static_cast<phi::GPUContext*>(pool.Get(place));
  kernel_run_stream_ = dev_ctx_->stream();
  cudaStreamCreateWithFlags(&load_stream_, cudaStreamNonBlocking);

  is_valied_ = true;
  buffer_nodes_.reserve(2);
  for (size_t i = 0; i < 2u; i++) {
    buffer_nodes_.push_back(Buffer(init_size));
  }

  weights_queue_ = weights_queue;
  weights_queue_cpy_ = weights_queue;
  sched_layers_ = sched_layers;

  for (auto& pair : weights_queue_) {
    CHECK_EQ(pair.second.first.size(), pair.second.second.size());
    for (size_t i = 0; i < pair.second.first.size(); i++) {
      CHECK_EQ(pair.second.first[i]->numel(), pair.second.second[i]->numel());
      CHECK_EQ(pair.second.first[i]->dtype(), pair.second.second[i]->dtype());
    }
  }
}

void SchedLayersPool::FillPool() {
  // fill free node
  for (size_t i = 0; i < buffer_nodes_.size(); i++) {
    if (weights_queue_.empty()) return;
    if (buffer_nodes_[i].free) {
      OpIdx2ParamsTensors op_params_info = weights_queue_.front();
      weights_queue_.pop_front();
      buffer_nodes_[i].layer_id = op_params_info.first;
      cudaStreamWaitEvent(load_stream_, buffer_nodes_[i].event, 0);
      for (size_t j = 0; j < op_params_info.second.first.size(); j++) {
        auto* src_tensor = op_params_info.second.first[j];
        auto* dst_tensor = op_params_info.second.second[j];
        dst_tensor->set_type(src_tensor->dtype());
        dst_tensor->Resize(src_tensor->dims());
        buffer_nodes_[i].gpu_resource.PartionMemoryTo(
            dst_tensor, dst_tensor->numel() * SizeOf(dst_tensor->dtype()));
        Transfer(*src_tensor,
                 dst_tensor,
                 src_tensor->numel() * SizeOf(src_tensor->dtype()),
                 load_stream_);
      }
      cudaEventRecord(buffer_nodes_[i].event, load_stream_);
      buffer_nodes_[i].free = false;
      active_layers_[buffer_nodes_[i].layer_id] = i;
    }
  }
}
}  // namespace framework
}  // namespace paddle
