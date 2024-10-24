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

template <typename T, typename ReturnType, typename... Args>
class KernelCallback {
 public:
  using ReturnT = ReturnType;
  using FuncType = ReturnType (*)(Args...);

  KernelCallback() {}
  explicit KernelCallback(FuncType f) : func(f) {}
  virtual ~KernelCallback() {}

  ReturnType Run(Args... args) { return func(args...); }

 private:
  FuncType func;
};

template <typename T, typename ReturnType, typename... Args>
static KernelCallback<T, ReturnType, Args...> MakeCallback(
    ReturnType (*cb)(Args...)) {
  return KernelCallback<T, ReturnType, Args...>(cb);
}

template <typename T, typename KernelType>
class AutoTuneBase {
 public:
  AutoTuneBase() {}
  virtual ~AutoTuneBase() {}

  explicit AutoTuneBase(KernelType default_kernel) {
    kernels_.push_back(default_kernel);
  }

  template <typename ReturnType, typename... Args>
  void AddCallBack(ReturnType (*func)(Args...)) {
    if (!is_init_) {
      std::lock_guard<std::mutex> lock(mutex_);
      kernels_.push_back(MakeCallback<T>(func));
    }
  }

  template <typename Context, typename... Args>
  void Run(const Context& ctx,
           const AlgorithmType& algo,
           const size_t key,
           Args&&... args) {
    is_init_ = true;
    CheckKernelSize();
    auto& cache = AutoTuneCache::Instance().Get(algo);
    if (cache.Find(key)) {
      auto best_idx = cache.Get(key);
      kernels_[best_idx].Run(args...);
    } else {
      bool use_autotune = AutoTuneStatus::Instance().UseAutoTune();
      if (use_autotune) {
        // All available kernels have ran while picking the best kernel,
        // so there may be no need for another kernel run.
        auto best_idx = PickBestKernel(ctx, args...);
        cache.Set(key, best_idx);
      } else {
        kernels_[0].Run(args...);
      }
    }
  }

 protected:
  bool is_init_{false};
  std::vector<KernelType> kernels_;
  mutable std::mutex mutex_;

  void CheckKernelSize() {
    PADDLE_ENFORCE_GT(
        kernels_.size(),
        0,
        common::errors::InvalidArgument(
            "kernel num must be greater than 0, now is %d", kernels_.size()));
  }

  template <typename Context, typename... Args>
  size_t PickBestKernel(const Context& ctx, Args&&... args) {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t best_idx = 0;
    float min_time = std::numeric_limits<float>::max();

    // Time cost test established in default stream.
    for (size_t i = 0; i < kernels_.size(); ++i) {
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
    // Regard 1st run as warmup, judge the compare result by the time cost
    // of rest cycles.
    constexpr int repeats = 11;
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

template <typename T, typename ReturnType, typename... Args>
class MatmulAutoTuner
    : public AutoTuneBase<T, KernelCallback<T, ReturnType, Args...>> {
 public:
  static MatmulAutoTuner<T, ReturnType, Args...>* Instance(
      ReturnType (*func)(Args...)) {
    static std::once_flag matmul_init_flag;
    static std::unique_ptr<MatmulAutoTuner<T, ReturnType, Args...>> instance;
    std::call_once(matmul_init_flag, [&] {
      auto obj = MakeCallback<T>(func);
      instance.reset(new MatmulAutoTuner<T, ReturnType, Args...>);
      instance->AddCallBack(func);
    });
    return instance.get();
  }

  template <typename Context>
  void Run(const Context& ctx, const size_t key, Args... args) {
    this->is_init_ = true;
    this->CheckKernelSize();
    auto& cache = AutoTuneCache::Instance().GetMatmul();
    if (cache.Find(key)) {
      auto best_idx = cache.Get(key);
      this->kernels_[best_idx].Run(args...);
    } else {
      bool use_autotune = AutoTuneStatus::Instance().UseAutoTune();
      if (use_autotune) {
        auto best_idx = this->PickBestKernel(ctx, args...);
        cache.Set(key, best_idx);
      } else {
        this->kernels_[0].Run(args...);
      }
    }
  }
};

template <bool TransposeA,
          bool TransposeB,
          typename T,
          typename ReturnType,
          typename... Args>
class GatherGemmScatterAutoTuner
    : public AutoTuneBase<T, KernelCallback<T, ReturnType, T, T, Args...>> {
 public:
  static GatherGemmScatterAutoTuner<TransposeA,
                                    TransposeB,
                                    T,
                                    ReturnType,
                                    Args...>*
  Instance(ReturnType (*func)(T, T, Args...)) {
    static std::once_flag gather_gemm_scatter_init_flag;
    static std::unique_ptr<GatherGemmScatterAutoTuner<TransposeA,
                                                      TransposeB,
                                                      T,
                                                      ReturnType,
                                                      Args...>>
        instance;
    std::call_once(gather_gemm_scatter_init_flag, [&] {
      auto obj = MakeCallback<T>(func);
      instance.reset(new GatherGemmScatterAutoTuner<TransposeA,
                                                    TransposeB,
                                                    T,
                                                    ReturnType,
                                                    Args...>);
      instance->AddCallBack(func);
    });
    return instance.get();
  }

  void Run(const phi::GPUContext& ctx,
           const size_t key,
           T const alpha,
           T const beta,
           Args... args) {
    this->is_init_ = true;
    this->CheckKernelSize();
    auto& cache = AutoTuneCache::Instance()
                      .GetGatherGemmScatter<T, TransposeA, TransposeB>();

    if (cache.Find(key)) {
      auto best_idx = cache.Get(key);
      this->kernels_[best_idx].Run(alpha, beta, args...);

    } else {
      // Set alpha to 0 and beta to 1 to avoid changing the value of d when
      // picking the best kernel
      auto best_idx =
          PickBestKernel(ctx, static_cast<T>(0), static_cast<T>(1), args...);
      cache.Set(key, best_idx);
      this->kernels_[best_idx].Run(alpha, beta, args...);
    }
  }

 protected:
  size_t PickBestKernel(const phi::GPUContext& ctx,
                        const T& alpha,
                        const T& beta,
                        Args&... args) {
    std::lock_guard<std::mutex> lock(this->mutex_);
    constexpr size_t NO_KERNEL_WORKS = -1;
    size_t best_idx = NO_KERNEL_WORKS;
    float min_time = std::numeric_limits<float>::max();

    // Time cost test established in default stream.
    for (int i = 0; i < this->kernels_.size(); ++i) {
      float time = 0;
      // Some kernels may require more shared memory than available, skip these
      // kernels.
      try {
        time = this->RunAndMeasureKernel(ctx, i, alpha, beta, args...);
        if (time < min_time) {
          min_time = time;
          best_idx = i;
        }
      } catch (const std::runtime_error& error) {
        VLOG(3) << "the kernels_[" << i << "] get error:" << error.what();
      }
    }
    if (best_idx == NO_KERNEL_WORKS) {
      LOG(ERROR) << "No kernel works!\n";
      exit(-1);
    }
    VLOG(3) << "best kernel idx is " << best_idx;
    return best_idx;
  }
};
template <bool TransposeA,
          bool TransposeB,
          typename T,
          typename ReturnType,
          typename... Args>
static GatherGemmScatterAutoTuner<TransposeA,
                                  TransposeB,
                                  T,
                                  ReturnType,
                                  Args...>*
MakeGatherGemmScatterTuner(ReturnType (*func)(T, T, Args...)) {
  return GatherGemmScatterAutoTuner<TransposeA,
                                    TransposeB,
                                    T,
                                    ReturnType,
                                    Args...>::Instance(func);
}

// Define the auto_tuner initial object.
#define DEFINE_AUTOTUNER_COMMON_OBJ(name)                                \
  template <typename T, typename ReturnType, typename... Args>           \
  class name##AutoTuner                                                  \
      : public AutoTuneBase<T, KernelCallback<T, ReturnType, Args...>> { \
   public:                                                               \
    static name##AutoTuner<T, ReturnType, Args...>* Instance(            \
        ReturnType (*func)(Args...)) {                                   \
      static std::once_flag name##_init_flag;                            \
      static std::unique_ptr<name##AutoTuner<T, ReturnType, Args...>>    \
          instance;                                                      \
      std::call_once(name##_init_flag, [&] {                             \
        auto obj = MakeCallback<T>(func);                                \
        instance.reset(new name##AutoTuner<T, ReturnType, Args...>);     \
        instance->AddCallBack(func);                                     \
      });                                                                \
      return instance.get();                                             \
    }                                                                    \
  };

// Define the auto_tuner initial function.
#define DEFINE_AUTOTUNER_FN(name)                                    \
  template <typename T, typename ReturnType, typename... Args>       \
  static name##AutoTuner<T, ReturnType, Args...>* Make##name##Tuner( \
      ReturnType (*func)(Args...)) {                                 \
    return name##AutoTuner<T, ReturnType, Args...>::Instance(func);  \
  }

#define DEFINE_AUTOTUNER(name)      \
  DEFINE_AUTOTUNER_COMMON_OBJ(name) \
  DEFINE_AUTOTUNER_FN(name)

DEFINE_AUTOTUNER(Transpose)
DEFINE_AUTOTUNER_FN(Matmul)

#undef DEFINE_AUTOTUNER_COMMON_OBJECT
#undef DEFINE_AUTOTUNER_FN
#undef DEFINE_AUTOTUNER

}  // namespace autotune
}  // namespace phi
