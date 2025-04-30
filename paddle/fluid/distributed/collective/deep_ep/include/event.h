// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include <glog/logging.h>
#include "paddle/fluid/distributed/collective/deep_ep/include/CUDAStream.h"
#include "paddle/fluid/distributed/collective/deep_ep/include/event_pool.h"
#include "paddle/fluid/distributed/collective/deep_ep/kernels/exception.cuh"

namespace deep_ep::detail {

class Event {
 public:
  Event() { cuda_event_ = EventPool::Instance().CreateCudaEventFromPool(); }
  void record(const cudaStream_t& stream) {
    CUDA_CHECK(cudaEventRecord(cuda_event_, stream));
  }

  cudaEvent_t cuda_event() const { return cuda_event_; }

 private:
  cudaEvent_t cuda_event_;
};

}  // namespace deep_ep::detail
