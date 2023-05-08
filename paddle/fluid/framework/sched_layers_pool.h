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

#include <list>

#include "paddle/fluid/platform/device_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/tensor_utils.h"

namespace paddle {

namespace framework {

using OpIdx2ParamsTensors = std::pair<
    size_t,
    std::pair<std::vector<phi::DenseTensor*>, std::vector<phi::DenseTensor*>>>;

struct ResourceHolder {
  ResourceHolder() {}

  void mutable_data(size_t required_size);

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
    free = true;
    gpu_resource.InitOffset();
  }

  ResourceHolder gpu_resource;
  size_t layer_id;
  bool free{true};
  cudaEvent_t event;
};

class SchedLayersPool {
 public:
  ~SchedLayersPool();

  void Init(size_t init_size,
            const std::list<OpIdx2ParamsTensors>& weights_queue,
            const std::vector<size_t>& sched_layers);

  bool IsValid() { return is_valied_; }

  void Transfer(const phi::DenseTensor& src,
                phi::DenseTensor* dst,
                size_t n,
                cudaStream_t stream);

  void FillPool();

  bool IsSchedLayer(size_t layer_idx) {
    auto find =
        std::find(sched_layers_.begin(), sched_layers_.end(), layer_idx);
    return find != sched_layers_.end();
  }

  void Debug() {
    LOG(INFO) << "buf id 0: free?: " << buffer_nodes_[0].free
              << ", layer_id: " << buffer_nodes_[0].layer_id;
    LOG(INFO) << "buf id 1: free?: " << buffer_nodes_[1].free
              << ", layer_id: " << buffer_nodes_[1].layer_id;
  }

  void Reset() { weights_queue_ = weights_queue_cpy_; }

 public:
  // layer id to buffer id
  std::map<size_t, size_t> active_layers_;
  std::vector<Buffer> buffer_nodes_;
  cudaStream_t kernel_run_stream_;

 private:
  std::list<OpIdx2ParamsTensors> weights_queue_;
  std::list<OpIdx2ParamsTensors> weights_queue_cpy_;
  std::vector<size_t> sched_layers_;
  cudaStream_t load_stream_;

  bool is_valied_{false};
};

class VectorSchedLayersPool {
 public:
  static VectorSchedLayersPool& Instance() {
    static VectorSchedLayersPool instance;
    return instance;
  }

  ~VectorSchedLayersPool() {
    for (auto* ptr : sched_layers_pools) delete ptr;
  }

  void Init(size_t idx,
            size_t init_size,
            const std::list<OpIdx2ParamsTensors>& weights_queue,
            const std::vector<size_t>& sched_layers) {
    sched_layers_pools[idx]->Init(init_size, weights_queue, sched_layers);
  }

  SchedLayersPool& Get(size_t idx) { return *sched_layers_pools[idx]; }

  size_t Size() { return sched_layers_pools.size(); }

 private:
  std::vector<SchedLayersPool*> sched_layers_pools;
  VectorSchedLayersPool() {
    sched_layers_pools.push_back(new SchedLayersPool());
    sched_layers_pools.push_back(new SchedLayersPool());
  }
};
}  // namespace framework
}  // namespace paddle
