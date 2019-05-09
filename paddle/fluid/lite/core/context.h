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

#include "paddle/fluid/lite/utils/any.h"
#ifdef LITE_WITH_CUDA
#include <paddle/fluid/lite/cuda/blas.h>
#include "paddle/fluid/lite/cuda/cuda_utils.h"
#endif
#include <memory>
#include <vector>
#include "paddle/fluid/lite/core/target_wrapper.h"

namespace paddle {
namespace lite {

struct HostContext {};

#ifdef LITE_WITH_ARM
struct ARMContext {};
#endif

#ifdef LITE_WITH_CUDA
// Only works with CUDA kernels.
struct CUDAContext {
  // overall information
  cudaStream_t exec_stream;
  cudaStream_t io_stream;

  // not thread-safe, should allocate for each thread.
  std::shared_ptr<cuda::Blas<float>> blas_fp32;

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
  template <typename ContextT>
  ContextT& As() {
    if (!ctx_.valid()) {
      ctx_.set<ContextT>();
    }
    return *ctx_.get_mutable<ContextT>();
  }

 private:
  Any ctx_;
};

}  // namespace lite
}  // namespace paddle
