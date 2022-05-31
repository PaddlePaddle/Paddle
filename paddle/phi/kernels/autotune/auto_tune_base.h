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

#include <mutex>
#include <type_traits>
#include "glog/logging.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/kernels/autotune/gpu_timer.h"

namespace phi {
namespace autotune {

template <typename T, typename RetureType, typename... Args>
class KernelCallback {
 public:
  using ReturnT = RetureType;
  using FuncType = RetureType (*)(Args...);

  KernelCallback() {}
  explicit KernelCallback(FuncType func_) : func(func_) {}
  virtual ~KernelCallback() {}

  RetureType Run(Args... args) { return func(args...); }

 private:
  FuncType func;
};

template <typename T, typename RetureType, typename... Args>
static KernelCallback<T, RetureType, Args...> MakeCallback(
    RetureType (*cb)(Args...)) {
  return KernelCallback<T, RetureType, Args...>(cb);
}

template <typename T, typename KernelType>
class AutoTuneBase {
 public:
  AutoTuneBase() {}
  virtual ~AutoTuneBase() {}
  explicit AutoTuneBase(KernelType kernel) { kernels_.push_back(kernel); }

  template <typename Type>
  void AddCallBack(Type kernel) {
    static_assert(std::is_same<Type, KernelType>::value,
                  "Type must be the same");
    kernels_.push_back(kernel);
  }

  template <typename... Args>
  void RunBestKernel(const int idx, Args&&... args) {
    KernelCall</*HasTimer=*/false>(idx, args...);
  }

  template <typename... Args>
  void RunDefaultKernel(Args&&... args) {
    KernelCall</*HasTimer=*/false>(0, args...);
  }

  template <typename Context, typename... Args>
  int PickBestKernel(const Context& ctx, Args&&... args) {
    PADDLE_ENFORCE_GT(
        kernels_.size(),
        0,
        paddle::platform::errors::InvalidArgument(
            "kernel num must be greater than 0, now is %d", kernels_.size()));

    float min_time = std::numeric_limits<float>::max();

    // Warm up step.
    ctx.Wait();
    KernelCall</*HasTimer=*/true>(0, args...);

    // Time cost test estabulished in default stream.
    for (int64_t i = 0; i < kernels_.size(); ++i) {
      auto time = KernelCall</*IsTune=*/true>(i, args...);
      VLOG(3) << "kernel[" << i << "] time cost is " << time / 2;

      if (time < min_time) {
        min_time = time;
        best_idx_ = i;
      }
    }
    VLOG(3) << "best kernel idx is " << best_idx_;
    return best_idx_;
  }

  bool CheckInit() { return is_init_; }
  void FinishInit() { is_init_ = true; }

 private:
  bool is_init_{false};
  int best_idx_{0};
  std::vector<KernelType> kernels_;

  template <bool IsTune, typename... Args>
  float KernelCall(const int idx, Args&&... args) {
    float time_cost = 0;

    if (IsTune) {
      phi::GpuTimer timer;
      for (int i = 0; i < /*repeat=*/2; ++i) {
        timer.Start(0);
        kernels_[idx].Run(args...);
        timer.Stop(0);
        time_cost += timer.ElapsedTime();
      }
    } else {
      kernels_[idx].Run(args...);
    }
    return time_cost;
  }
};

template <typename T, typename RetureType, typename... Args>
static AutoTuneBase<T, KernelCallback<T, RetureType, Args...>> MakeAutoTuner(
    RetureType (*func)(Args...)) {
  auto obj = MakeCallback<T>(func);
  return AutoTuneBase<T, decltype(obj)>(obj);
}

template <typename T, typename KernelType>
class TransposeTuneHelper : public AutoTuneBase<T, KernelType> {
 public:
  static AutoTuneBase<T, KernelType>* Instance(KernelType kernel) {
    static std::unique_ptr<AutoTuneBase<T, KernelType>> instance_;
    std::call_once(init_flag_, [&] {
      instance_.reset(new AutoTuneBase<T, KernelType>(kernel));
    });
    return instance_.get();
  }

 private:
  static std::once_flag init_flag_;
};

template <typename T, typename KernelType>
std::once_flag TransposeTuneHelper<T, KernelType>::init_flag_;

template <typename T, typename RetureType, typename... Args>
static AutoTuneBase<T, KernelCallback<T, RetureType, Args...>>*
    MakeTransposeTuner(RetureType (*func)(Args...)) {
  auto obj = MakeCallback<T>(func);
  return TransposeTuneHelper<T, decltype(obj)>::Instance(obj);
}

}  // namespace autotune
}  // namespace phi
