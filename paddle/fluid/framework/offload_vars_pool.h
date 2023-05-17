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

#pragma once

#ifdef PADDLE_WITH_CUDA
#include <list>

#include "paddle/fluid/platform/device_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/tensor_utils.h"

namespace paddle {

namespace framework {

using LayerIdx2ParamsTensors = std::pair<
    size_t,
    std::pair<std::vector<phi::DenseTensor*>, std::vector<phi::DenseTensor*>>>;

struct ResourceHolder {
  void Alloc(size_t required_size);

  void InitOffset() { tensor_.set_offset(0); }

  void PartionMemoryTo(phi::DenseTensor* tensor, size_t byte_size) {
    auto& holder = phi::DenseTensorUtils::GetHolder(tensor_);
    tensor->set_offset(tensor_.offset());
    tensor->ResetHolder(holder);
    tensor_.set_offset(tensor_.offset() + byte_size);
  }
  phi::DenseTensor tensor_;
};

struct Buffer {
  explicit Buffer(size_t required_size);

  ~Buffer();

  void ResetBuffer() {
    free_ = true;
    gpu_resource_.InitOffset();
  }

  ResourceHolder gpu_resource_;
  size_t layer_id_;
  bool free_{true};
  cudaEvent_t event_;
};

class OffloadVarsPool {
 public:
  ~OffloadVarsPool();

  void Init(size_t init_size,
            const std::list<LayerIdx2ParamsTensors>& weights_queue,
            const std::unordered_set<size_t>& sched_layers);

  void Transfer(const phi::DenseTensor& src,
                phi::DenseTensor* dst,
                size_t n,
                cudaStream_t stream);

  void FillPool();

  bool IsOffloadLayer(size_t layer_idx) {
    auto find =
        std::find(offload_layers_.begin(), offload_layers_.end(), layer_idx);
    return find != offload_layers_.end();
  }

  void Reset() {
    weights_queue_ = weights_queue_cpy_;
    FillPool();
  }

  void WaitCopyCompleted(size_t layer_idx) {
    size_t buf_id = active_layers_[layer_idx];
    cudaStreamWaitEvent(kernel_run_stream_, buffer_nodes_[buf_id].event_, 0);
  }

  void RecordEvent(size_t layer_idx) {
    size_t buf_id = active_layers_[layer_idx];
    cudaEventRecord(buffer_nodes_[buf_id].event_, kernel_run_stream_);
    buffer_nodes_[buf_id].ResetBuffer();
  }

 public:
  // layer id to buffer id
  std::map<size_t, size_t> active_layers_;
  std::vector<Buffer> buffer_nodes_;
  cudaStream_t kernel_run_stream_;

 private:
  std::list<LayerIdx2ParamsTensors> weights_queue_;
  std::list<LayerIdx2ParamsTensors> weights_queue_cpy_;
  std::unordered_set<size_t> offload_layers_;
  cudaStream_t load_stream_;
};

class OffloadVarsPoolVector {
 public:
  static OffloadVarsPoolVector& Instance() {
    static OffloadVarsPoolVector instance;
    return instance;
  }

  ~OffloadVarsPoolVector() {
    for (auto* ptr : offload_vars_pools) delete ptr;
  }

  void Init(size_t idx,
            size_t init_size,
            const std::list<LayerIdx2ParamsTensors>& weights_queue,
            const std::unordered_set<size_t>& offload_layers) {
    if (Size() <= idx) offload_vars_pools.push_back(new OffloadVarsPool());

    PADDLE_ENFORCE_GT(
        Size(),
        idx,
        platform::errors::InvalidArgument(
            "size of offload_layers_pools must be greater than "
            "idx, but idx is %d, and size of offload_layers_pools is %d.",
            idx,
            Size()));

    offload_vars_pools[idx]->Init(init_size, weights_queue, offload_layers);
  }

  OffloadVarsPool& Get(size_t idx) { return *offload_vars_pools[idx]; }

  size_t Size() { return offload_vars_pools.size(); }

 private:
  std::vector<OffloadVarsPool*> offload_vars_pools;
  OffloadVarsPoolVector() {
    offload_vars_pools.push_back(new OffloadVarsPool());
  }
};
}  // namespace framework
}  // namespace paddle
#endif
