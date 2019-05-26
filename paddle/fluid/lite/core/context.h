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
#include "paddle/fluid/lite/cuda/blas.h"
#include "paddle/fluid/lite/cuda/cuda_utils.h"
#endif
#ifdef LITE_WITH_X86
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/device_context.h"
#endif
#include <memory>
#include <set>
#include <vector>
#include "paddle/fluid/lite/core/cpu_info.h"
#include "paddle/fluid/lite/core/lite_tensor.h"
#include "paddle/fluid/lite/core/target_wrapper.h"

namespace paddle {
namespace lite {

struct HostContext {};

#ifdef LITE_WITH_ARM

struct ARMContext {
 public:
  ARMContext();
  ARMContext(PowerMode mode, int threads);
  ARMContext(const ARMContext& ctx);

  ARMContext& operator=(const ARMContext& ctx);

  void set_run_mode(PowerMode mode, int threads);
  void bind_dev();
  PowerMode get_mode() const;
  int get_threads() const;
  void set_cache(int l1size, int l2size, int l3size);
  template <typename T>
  T* get_workspace_data() {
    return workspace_.mutable_data<T>();
  }
  ARMArch get_arch() const;
  void set_arch(ARMArch arch);
  int l1_cache_size() const;
  int l2_cache_size() const;
  int l3_cache_size() const;
  bool workspace_extend(DDimLite dims);

 private:
  //! LITE_POWER_HIGH stands for using big cores,
  //! LITE_POWER_LOW stands for using small core,
  //! LITE_POWER_FULL stands for using all cores
  ARMArch arch_;
  PowerMode mode_;
  std::vector<int> act_ids_;
  TensorLite workspace_;
  int64_t count_{0};
};
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

  // legacy info.
  std::unique_ptr<::paddle::platform::CPUDeviceContext> x86_device_context;
  std::unique_ptr<::paddle::framework::ExecutionContext> x86_execution_context;
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
