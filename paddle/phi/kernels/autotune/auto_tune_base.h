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

#include <type_traits>
#include "glog/logging.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/kernels/autotune/gpu_timer.h"

namespace phi {
namespace autotune {

template <typename RetureType, typename... Args>
class KernelCallback {
 public:
  using ReturnT = RetureType;
  using FuncType = RetureType (*)(Args...);

  KernelCallback() {}
  explicit KernelCallback(FuncType func_) : func(func_) {}
  virtual ~KernelCallback() {}

  RetureType Call(Args... args) { return func(args...); }

 private:
  FuncType func;
};

template <typename RetureType, typename... Args>
static KernelCallback<RetureType, Args...> MakeCallback(
    RetureType (*cb)(Args...)) {
  return KernelCallback<RetureType, Args...>(cb);
}

template <typename KernelType>
class AutoTuneBase {
 public:
  AutoTuneBase() {}
  virtual ~AutoTuneBase() {}
  explicit AutoTuneBase(KernelType kernel) : default_kernel_(kernel) {
    kernels_.push_back(kernel);
  }

  template <typename T>
  void AddCallBack(T kernel) {
    static_assert(std::is_same<T, KernelType>::value, "Type must be the same");
    kernels_.push_back(kernel);
  }

  template <typename Context, typename... Args>
  KernelType PickBestKernel(const Context& ctx, Args&&... args) {
    PADDLE_ENFORCE_GT(
        kernels_.size(),
        0,
        paddle::platform::errors::InvalidArgument(
            "kernel num must be greater than 0, now is %d", kernels_.size()));
    int idx = 0;
    phi::GpuTimer timer;
    float min_time = std::numeric_limits<float>::max();

    for (int i = 0; i < kernels_.size(); ++i) {
      ctx.Wait();
      timer.Start(0);
      kernels_[i].Call(args...);
      timer.Stop(0);
      auto time = timer.ElapsedTime();
      VLOG(3) << "kernel[" << i << "]: time cost is " << time;

      if (time < min_time) {
        min_time = time;
        idx = i;
      }
    }
    VLOG(3) << "best kernel idx is " << idx;
    return kernels_[idx];
  }

 private:
  KernelType default_kernel_;
  std::vector<KernelType> kernels_;
};

template <typename RetureType, typename... Args>
static AutoTuneBase<KernelCallback<RetureType, Args...>> MakeAutoTuner(
    RetureType (*func)(Args...)) {
  auto obj = MakeCallback(func);
  return AutoTuneBase<decltype(obj)>(obj);
}

}  // namespace autotune
}  // namespace phi
