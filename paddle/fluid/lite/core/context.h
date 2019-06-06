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
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include "paddle/fluid/lite/core/cpu_info.h"
#include "paddle/fluid/lite/core/lite_tensor.h"
#include "paddle/fluid/lite/core/target_wrapper.h"
#include "paddle/fluid/lite/utils/all.h"

namespace paddle {
namespace lite {

template <TargetType Type>
class Context;

using HostContext = Context<TargetType::kHost>;
using X86Context = Context<TargetType::kX86>;
using CUDAContext = Context<TargetType::kCUDA>;
using ARMContext = Context<TargetType::kARM>;

template <>
class Context<TargetType::kHost> {
 public:
  // NOTE: InitOnce should only be used by ContextScheduler
  void InitOnce() {}

  void CopySharedFrom(const HostContext& ctx) {}

  std::string name() const { return "HostContext"; }
};

#ifdef LITE_WITH_ARM

template <>
class Context<TargetType::kARM> {
 public:
  Context();
  Context(PowerMode mode, int threads);
  explicit Context(const ARMContext& ctx);

  ARMContext& operator=(const ARMContext& ctx);

  // NOTE: InitOnce should only be used by ContextScheduler
  void InitOnce() { DeviceInfo::Init(); }

  void CopySharedFrom(const ARMContext& ctx) {}

  void SetRunMode(PowerMode mode, int threads);
  void SetCache(int l1size, int l2size, int l3size);
  void SetArch(ARMArch arch);
  void BindDev();

  PowerMode mode() const;
  int threads() const;
  ARMArch arch() const;

  template <typename T>
  T* workspace_data() {
    return workspace_.mutable_data<T>();
  }

  int l1_cache_size() const;
  int l2_cache_size() const;
  int l3_cache_size() const;
  bool ExtendWorkspace(DDimLite dims);

  std::string name() const { return "ARMContext"; }

 private:
  // LITE_POWER_HIGH stands for using big cores,
  // LITE_POWER_LOW stands for using small core,
  // LITE_POWER_FULL stands for using all cores
  ARMArch arch_;
  PowerMode mode_;
  std::vector<int> active_ids_;
  TensorLite workspace_;
  int64_t count_{0};
};
#endif

#ifdef LITE_WITH_CUDA
// Only works with CUDA kernels.
template <>
class Context<TargetType::kCUDA> {
 public:
  // NOTE: InitOnce should only be used by ContextScheduler
  void InitOnce() {
    cublas_fp32_ = std::make_shared<lite::cuda::Blas<float>>();
  }

  void CopySharedFrom(const CUDAContext& ctx) {
    CHECK(ctx.cublas_fp32()) << "cublas_fp32 should be set first";
    cublas_fp32_ = ctx.cublas_fp32();
  }

  const cudaStream_t exec_stream() { return exec_stream_; }
  void SetExecStream(cudaStream_t stream) { exec_stream_ = stream; }

  const cudaStream_t io_stream() { return io_stream_; }
  void SetIoStream(cudaStream_t stream) { io_stream_ = stream; }

  std::shared_ptr<cuda::Blas<float>> cublas_fp32() { return cublas_fp32_; }
  void SetCuBlasFP32(std::shared_ptr<cuda::Blas<float>> cublas_fp32) {
    cublas_fp32_ = cublas_fp32;
  }

  const std::vector<cudaEvent_t>& input_events() { return input_events_; }
  void SetInputEvents(const std::vector<cudaEvent_t>& input_events) {
    input_events_.clear();
    input_events_.assign(input_events.begin(), input_events.end());
  }

  const std::vector<cudaEvent_t>& output_events() { return output_events_; }
  void SetOutputEvents(const std::vector<cudaEvent_t>& output_events) {
    output_events_.clear();
    output_events_.assign(output_events.begin(), output_events.end());
  }

  std::string name() const { return "CUDAContext"; }

 private:
  // overall information
  cudaStream_t exec_stream_;
  cudaStream_t io_stream_;

  // not thread-safe, should allocate for each thread.
  std::shared_ptr<cuda::Blas<float>> cublas_fp32_;

  // kernel information
  std::vector<cudaEvent_t> input_events_;
  std::vector<cudaEvent_t> output_events_;
};
#endif

#ifdef LITE_WITH_X86
template <>
class Context<TargetType::kX86> {
 public:
  using device_ctx_t = ::paddle::platform::CPUDeviceContext;
  using execution_ctx_t = ::paddle::framework::ExecutionContext;

  Context() {
    x86_device_context_.reset(new ::paddle::platform::CPUDeviceContext);
    x86_execution_context_.reset(
        new ::paddle::framework::ExecutionContext(*x86_device_context_));
  }

  // NOTE: InitOnce should only be used by ContextScheduler
  void InitOnce() {}

  void CopySharedFrom(const X86Context& ctx) {}

  const device_ctx_t* x86_device_context() { return x86_device_context_.get(); }
  void SetX86DeviceContext(std::unique_ptr<device_ctx_t>&& ctx) {
    x86_device_context_ = std::move(ctx);
  }

  const execution_ctx_t* x86_execution_context() {
    return x86_execution_context_.get();
  }
  void SetX86ExecutionContext(std::unique_ptr<execution_ctx_t>&& ctx) {
    x86_execution_context_ = std::move(ctx);
  }

  std::string name() const { return "X86Context"; }

 private:
  // overall information
  //
  // kernel information

  // legacy info.
  std::unique_ptr<device_ctx_t> x86_device_context_;
  std::unique_ptr<execution_ctx_t> x86_execution_context_;
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

// The ContextScheduler helps to assign different context for each kernel.
class ContextScheduler {
 public:
  static ContextScheduler& Global() {
    static auto* x = new ContextScheduler;
    return *x;
  }

  std::unique_ptr<KernelContext> NewContext(TargetType target) {
    std::unique_ptr<KernelContext> ctx(new KernelContext);
    switch (target) {
      case TARGET(kHost):
        ctx->As<HostContext>().CopySharedFrom(
            kernel_contexts_[TargetType::kHost].As<HostContext>());
        break;
#ifdef LITE_WITH_X86
      case TARGET(kX86):
        ctx->As<X86Context>().CopySharedFrom(
            kernel_contexts_[TargetType::kX86].As<X86Context>());
        break;
#endif
#ifdef LITE_WITH_CUDA
      case TARGET(kCUDA):
        ctx->As<CUDAContext>().CopySharedFrom(
            kernel_contexts_[TargetType::kCUDA].As<CUDAContext>());
        break;
#endif
#ifdef LITE_WITH_ARM
      case TARGET(kARM):
        ctx->As<ARMContext>().CopySharedFrom(
            kernel_contexts_[TargetType::kARM].As<ARMContext>());
        break;
#endif
      default:
        LOG(FATAL) << "unsupported target " << TargetToStr(target);
    }
    return ctx;
  }

 private:
  template <TargetType Type, typename ContextT>
  void InitContext() {
    kernel_contexts_[Type].As<ContextT>().InitOnce();
  }

  ContextScheduler() {
    InitContext<TargetType::kHost, HostContext>();
#ifdef LITE_WITH_X86
    InitContext<TargetType::kX86, X86Context>();
#endif
#ifdef LITE_WITH_CUDA
    InitContext<TargetType::kCUDA, CUDAContext>();
#endif
#ifdef LITE_WITH_ARM
    InitContext<TargetType::kARM, ARMContext>();
#endif
  }

 private:
  std::map<TargetType, KernelContext> kernel_contexts_;
};

}  // namespace lite
}  // namespace paddle
