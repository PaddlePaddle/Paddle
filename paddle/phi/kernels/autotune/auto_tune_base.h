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
#include "paddle/phi/kernels/autotune/gpu_timer.h"
#include "paddle/phi/kernels/autotune/switch_autotune.h"

namespace phi {
namespace autotune {

<<<<<<< HEAD
template <typename T, typename ReturnType, typename... Args>
class KernelCallback {
 public:
  using ReturnT = ReturnType;
  using FuncType = ReturnType (*)(Args...);
=======
template <typename T, typename RetureType, typename... Args>
class KernelCallback {
 public:
  using ReturnT = RetureType;
  using FuncType = RetureType (*)(Args...);
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

  KernelCallback() {}
  explicit KernelCallback(FuncType func_) : func(func_) {}
  virtual ~KernelCallback() {}

<<<<<<< HEAD
  ReturnType Run(Args... args) { return func(args...); }
=======
  RetureType Run(Args... args) { return func(args...); }
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

 private:
  FuncType func;
};

<<<<<<< HEAD
template <typename T, typename ReturnType, typename... Args>
static KernelCallback<T, ReturnType, Args...> MakeCallback(
    ReturnType (*cb)(Args...)) {
  return KernelCallback<T, ReturnType, Args...>(cb);
=======
template <typename T, typename RetureType, typename... Args>
static KernelCallback<T, RetureType, Args...> MakeCallback(
    RetureType (*cb)(Args...)) {
  return KernelCallback<T, RetureType, Args...>(cb);
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
}

template <typename T, typename KernelType>
class AutoTuneBase {
 public:
  AutoTuneBase() {}
  virtual ~AutoTuneBase() {}

  explicit AutoTuneBase(KernelType kernel) {
    kernels_.push_back(/*default=*/kernel);
  }

<<<<<<< HEAD
  template <typename ReturnType, typename... Args>
  void AddCallBack(ReturnType (*func)(Args...)) {
    if (!is_init_) {
      std::lock_guard<std::mutex> lock(mutex_);
      kernels_.push_back(MakeCallback<T>(func));
=======
  void AddCallBack(KernelType kernel) {
    if (!is_init_) {
      std::lock_guard<std::mutex> lock(mutex_);
      kernels_.push_back(kernel);
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    }
  }

  template <typename Context, typename... Args>
  void Run(const Context& ctx,
           const AlgorithmType& algo,
           const size_t key,
           Args&&... args) {
    PADDLE_ENFORCE_GT(
        kernels_.size(),
        0,
<<<<<<< HEAD
        phi::errors::InvalidArgument(
=======
        paddle::platform::errors::InvalidArgument(
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            "kernel num must be greater than 0, now is %d", kernels_.size()));
    is_init_ = true;

    auto& cache = AutoTuneCache::Instance().Get(algo);
    if (cache.Find(key)) {
      auto best_idx = cache.Get(key);
      kernels_[best_idx].Run(args...);
    } else {
      bool use_autotune = AutoTuneStatus::Instance().UseAutoTune();
      if (use_autotune) {
        // All avaliable kernels have ran while picking the best kernel,
        // so there may be no need for another kernel run.
        auto best_idx = PickBestKernel(ctx, args...);
        cache.Set(key, best_idx);
      } else {
        kernels_[0].Run(args...);
      }
    }
  }

 private:
  bool is_init_{false};
  std::vector<KernelType> kernels_;
  mutable std::mutex mutex_;

  template <typename Context, typename... Args>
  size_t PickBestKernel(const Context& ctx, Args&&... args) {
    std::lock_guard<std::mutex> lock(mutex_);
    PADDLE_ENFORCE_GT(
        kernels_.size(),
        0,
<<<<<<< HEAD
        phi::errors::InvalidArgument(
=======
        paddle::platform::errors::InvalidArgument(
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            "kernel num must be greater than 0, now is %d", kernels_.size()));
    size_t best_idx = 0;
    float min_time = std::numeric_limits<float>::max();

    // Time cost test estabulished in default stream.
    for (int i = 0; i < kernels_.size(); ++i) {
      auto time = RunAndMeasureKernel<Context>(ctx, i, args...);
      if (time < min_time) {
        min_time = time;
        best_idx = i;
      }
    }
    VLOG(3) << "best kernel idx is " << best_idx;
    return best_idx;
  }

  template <typename Context, typename... Args>
  float RunAndMeasureKernel(const Context& ctx, const int idx, Args&&... args) {
<<<<<<< HEAD
    // Regard 1st run as warmup, judge the compare result by the time cost
    // of rest cycles.
    constexpr int repeats = 6;
=======
    // Regard 1st run as warmup. Judge the result by the time cost of rest run
    // cycles.
    constexpr int repeats = 3;
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    phi::GpuTimer timer;
    float time_cost = 0;
    const auto& stream = ctx.stream();

    ctx.Wait();
    for (int i = 0; i < repeats; ++i) {
      timer.Start(stream);
      kernels_[idx].Run(args...);
      timer.Stop(stream);
      auto time = timer.ElapsedTime();
      if (i > 0) {
        time_cost += time;
      }
      VLOG(3) << "kernel[" << idx << "][" << i << "th time cost is " << time;
    }
    return time_cost;
  }
};

<<<<<<< HEAD
template <typename T, typename ReturnType, typename... Args>
static AutoTuneBase<T, KernelCallback<T, ReturnType, Args...>> MakeAutoTuner(
    ReturnType (*func)(Args...)) {
=======
template <typename T, typename RetureType, typename... Args>
static AutoTuneBase<T, KernelCallback<T, RetureType, Args...>> MakeAutoTuner(
    RetureType (*func)(Args...)) {
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
  auto obj = MakeCallback<T>(func);
  return AutoTuneBase<T, decltype(obj)>(obj);
}

<<<<<<< HEAD
template <typename T, typename ReturnType, typename... Args>
class TransposeAutoTuner
    : public AutoTuneBase<T, KernelCallback<T, ReturnType, Args...>> {
 public:
  static AutoTuneBase<T, KernelCallback<T, ReturnType, Args...>>* Instance(
      ReturnType (*func)(Args...)) {
    static std::once_flag transpose_init_flag_;
    static std::unique_ptr<
        AutoTuneBase<T, KernelCallback<T, ReturnType, Args...>>>
        instance_;
    std::call_once(transpose_init_flag_, [&] {
      auto obj = MakeCallback<T>(func);
      instance_.reset(new AutoTuneBase<T, decltype(obj)>(obj));
    });
    return instance_.get();
  }
};

template <typename T, typename ReturnType, typename... Args>
static AutoTuneBase<T, KernelCallback<T, ReturnType, Args...>>*
MakeTransposeTuner(ReturnType (*func)(Args...)) {
  return TransposeAutoTuner<T, ReturnType, Args...>::Instance(func);
=======
template <typename T, typename KernelType>
class TransposeAutoTuner : public AutoTuneBase<T, KernelType> {
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
std::once_flag TransposeAutoTuner<T, KernelType>::init_flag_;

template <typename T, typename RetureType, typename... Args>
static AutoTuneBase<T, KernelCallback<T, RetureType, Args...>>*
MakeTransposeTuner(RetureType (*func)(Args...)) {
  auto obj = MakeCallback<T>(func);
  return TransposeAutoTuner<T, decltype(obj)>::Instance(obj);
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
}

}  // namespace autotune
}  // namespace phi
