// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
#include <paddle/fluid/lite/cuda/blas.h>
#include <memory>
#include <vector>
#include "paddle/fluid/lite/core/target_wrapper.h"

#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/lite/cuda/cuda_utils.h"
#endif

namespace paddle {
namespace lite {

template <TargetType Target>
class Context {
 public:
  using target_wrapper_t = TargetWrapper<Target>;
  using stream_t = typename TargetWrapper<Target>::stream_t;
  using event_t = typename TargetWrapper<Target>::event_t;

  Context() = default;
  Context(int device_id, stream_t compute_stream, stream_t data_stream)
      : device_id_(device_id),
        compute_stream_(compute_stream),
        data_stream_(data_stream) {}

  void SetDeviceId(int device_id) { device_id_ = device_id; }
  void SetComputeStream(stream_t x) { compute_stream_ = x; }
  void SetDataStream(stream_t x) { data_stream_ = x; }
  void SetDependEvents(const std::vector<event_t>& events) {
    depend_events_ = events;
  }

  int device_id() const { return device_id_; }
  stream_t compute_stream() const { return compute_stream_; }
  stream_t data_stream() const { return data_stream_; }
  const std::vector<event_t>& depend_events() const { return depend_events_; }

 private:
  int device_id_{0};
  stream_t compute_stream_;
  stream_t data_stream_;
  std::vector<event_t> depend_events_;
};

class OpContext final {
 public:
  template <TargetType Target>
  using target_ptr_t = std::unique_ptr<Context<Target>>;

  // @param target valid target.
  explicit OpContext(TargetType target)
      : targets_(std::vector<TargetType>({target})) {}
  // @param target valid target.
  explicit OpContext(const std::vector<TargetType>& target)
      : targets_(target) {}

  const std::vector<TargetType>& target() const { return targets_; }

  template <TargetType Target>
  target_ptr_t<Target> CreateContext() {
    return target_ptr_t<Target>(new Context<Target>);
  }

 private:
  std::vector<TargetType> targets_;
};

#ifdef LITE_WITH_CUDA
// Only works with CUDA kernels.
struct CUDAContext {
  // overall information
  cudaStream_t exec_stream;
  cudaStream_t io_stream;

  // not thread-safe, should allocate for each thread.
  std::shared_ptr<cuda::Blas<float>> bias_fp32;

  // kernel information
  std::vector<cudaEvent_t> input_events;
  std::vector<cudaEvent_t> output_events;
};
#endif

#ifdef LITE_WITH_X86
struct X86Context {
  // overall information
  // kernel information
};
#endif

// Context for running a kernel.
// Holds the necessary resource and information.
class KernelContext {
 public:
#ifdef LITE_WITH_CUDA
  CUDAContext cuda_ctx;
#endif

#ifdef LITE_WITH_X86
  X86Context x86_ctx;
#endif
};

}  // namespace lite
}  // namespace paddle
