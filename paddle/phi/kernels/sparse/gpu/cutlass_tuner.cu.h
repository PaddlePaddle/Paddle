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

#include "paddle/phi/common/data_type.h"
#include "paddle/phi/kernels/autotune/cache.h"
#ifdef PADDLE_WITH_CUTLASS
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_factory.h"
#include "paddle/phi/kernels/autotune/auto_tune_base.h"
#include "paddle/phi/kernels/autotune/cache_base.h"
namespace phi {
namespace sparse {
struct CutlassCacheKey {
  CutlassCacheKey() {}
  CutlassCacheKey(const int arg_m,
                  const int arg_n,
                  const int arg_k,
                  const bool arg_transposedA,
                  const bool arg_transposedB,
                  phi::DataType arg_dtype)
      : m(arg_m),
        n(arg_n),
        k(arg_k),
        transposedA(arg_transposedA),
        transposedB(arg_transposedB),
        dtype(arg_dtype) {}
  size_t hash_value() const {
    return autotune::GetKey(m, n, k, transposedA, transposedB, dtype);
  }
  int m;
  int n;
  int k;
  bool transposedA;
  bool transposedB;
  phi::DataType dtype;
};
struct CutlassCacheKeyHash {
  size_t operator()(const CutlassCacheKey& cache) const {
    return cache.hash_value();
  }
};
struct CutlassCacheKeyEqual {
  size_t operator()(const CutlassCacheKey& first,
                    const CutlassCacheKey& second) const {
    if (first.m != second.m) return false;
    if (first.n != second.n) return false;
    if (first.k != second.k) return false;
    if (first.transposedA != second.transposedB) return false;
    if (first.dtype != second.dtype) return false;
    return true;
  }
};
template <typename AlgorithmT>
class CutlassAlgorithmCache
    : public autotune::AlgorithmsCache<CutlassCacheKey,
                                       AlgorithmT,
                                       CutlassCacheKeyHash,
                                       CutlassCacheKeyEqual> {
 public:
  using AlgorithmsCacheBase = autotune::AlgorithmsCache<CutlassCacheKey,
                                                        AlgorithmT,
                                                        CutlassCacheKeyHash,
                                                        CutlassCacheKeyEqual>;
  CutlassAlgorithmCache()
      : phi::autotune::AlgorithmsCache<CutlassCacheKey,
                                       AlgorithmT,
                                       CutlassCacheKeyHash,
                                       CutlassCacheKeyEqual>() {}
  void Set(const CutlassCacheKey& key, AlgorithmT algo) {
    std::lock_guard<std::mutex> lock(*AlgorithmsCacheBase::cache_mutex_);
    if (AlgorithmsCacheBase::hash_.size() >
        static_cast<size_t>(FLAGS_search_cache_max_number)) {
      AlgorithmsCacheBase::hash_.clear();
    }
    AlgorithmsCacheBase::hash_[key] = algo;
  }
};

#if 0
  /// it seems that we dont need this.
template <typename AlgorithmT>
class CutlassAutoTuneCache : public autotune::AutoTuneCache {
 public:
  using AlgorithmsCacheBase = autotune::AlgorithmsCache<CutlassCacheKey,
                                                        AlgorithmT,
                                                        CutlassCacheKeyHash,
                                                        CutlassCacheKeyEqual>;
};
#endif
template <typename T, typename ReturnType, typename... Args>
class CutlassAutoTuner
    : public autotune::
          AutoTuneBase<T, autotune::KernelCallback<T, ReturnType, Args...>> {
 public:
  static autotune::
      AutoTuneBase<T, autotune::KernelCallback<T, ReturnType, Args...>>*
      Instance(ReturnType (*func)(Args...)) {
    static std::once_flag cutlass_init_flag_;
    static std::unique_ptr<autotune::AutoTuneBase<
        T,
        autotune::KernelCallback<T, ReturnType, Args...>>>
        instance_;
    std::call_once(cutlass_init_flag_, [&] {
      auto obj = autotune::MakeCallback<T>(func);
      instance_.reset(new autotune::AutoTuneBase<T, decltype(obj)>(obj));
    });
    return instance_.get();
  }

  template <typename Context, typename... Args>
  void Run(const Context& ctx, const size_t key, Args&&... args) {
    PADDLE_ENFORCE_GT(
        kernels_.size(),
        0,
        phi::errors::InvalidArgument(
            "kernel num must be greater than 0, now is %d", kernels_.size()));
    is_init_ = true;

    auto& cache = autotune::AutoTuneCache::Instance().Get(
        autotune::AlgorithmType::kCutlass);
    if (cache.Find(key)) {
      auto best_idx = cache.Get(key);
      kernels_[best_idx].Run(args...);
    } else {
      bool use_autotune = autotune::AutoTuneStatus::Instance().UseAutoTune();
      // All avaliable kernels have ran while picking the best kernel,
      // so there may be no need for another kernel run.
      auto best_idx = PickBestKernel(ctx, args...);
      cache.Set(key, best_idx);
    }
  }
  template <typename Context, typename... Args>
  size_t PickBestKernel(const Context& ctx, Args&&... args) {
    std::lock_guard<std::mutex> lock(mutex_);
    PADDLE_ENFORCE_GT(
        kernels_.size(),
        0,
        phi::errors::InvalidArgument(
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

 private:
};
size_t Conv3DKey(const int m, const int n, const int k) {
  return autotune::GetKey(m, n, k);
}

template <typename T, typename ReturnType, typename... Args>
static autotune::AutoTuneBase<T,
                              autotune::KernelCallback<T, ReturnType, Args...>>*
MakeCutlassTuner(ReturnType (*func)(Args...)) {
  return CutlassAutoTuner<T, ReturnType, Args...>::Instance(func);
}
#if 0
    // do we really need a pre-defined rules?
    struct GatherGemmScatterSimple {
      template <typename T>
      static bool Impl(const phi::GPUContext& ctx,
		       const T* const a,
		       const T* const bb,
		       const T* const c,
		       T* const d,
		       const int m,
		       const int n,
		       const int k,
		       const int32_t* a_indices,
		       const int32_t* b_indices,
		       const int32_t* c_d_indices,
		       T const alpha,
		       T const beta) {
	
      }
    private:
      template <typename T>
      static bool Run(const phi::GPUContext& ctx,
		      const T* const a,
		      const T* const b,
		      const T* const c,
		      T* const d,
		      const int m,
		      const int n,
		      const int k,
		      const int32_t* a_indices,
		      const int32_t* b_indices,
		      const int32_t* c_d_indices,
		      T const alpha,
		      T const beta) {
	// define some rules offline here
	return false;	
      }
    }
#endif
template <typename T>
struct GatherGemmScatterSimple {
  static bool Impl(const phi::GPUContext& ctx,
                   const T* const a,
                   const T* const b,
                   const T* const c,
                   T* const d,
                   const int m,
                   const int n,
                   const int k,
                   const int32_t* a_indices,
                   const int32_t* b_indices,
                   const int32_t* c_d_indices,
                   T const alpha,
                   T const beta) {
    return false;
#if 0
      return Run(ctx,
                          a,
                          b,
                          c,
                          d,
                          m,
                          n,
                          k,
                          a_indices,
                          b_indices,
                          c_d_indices,
                          alpha,
                          bet);
#endif
  }

 private:
  static bool Run(const phi::GPUContext& ctx,
                  const T* const a,
                  const T* const b,
                  const T* const c,
                  T* const d,
                  const int m,
                  const int n,
                  const int k,
                  const int32_t* a_indices,
                  const int32_t* b_indices,
                  const int32_t* c_d_indices,
                  T const alpha,
                  T const beta) {
      return false;
  }
};

template <typename T>
void GatherGemmScatterGPUKernelDriver(const phi::GPUContext& ctx,
                                      const T* const a,
                                      const T* const b,
                                      const T* const c,
                                      T* const d,
                                      const int m,
                                      const int n,
                                      const int k,
                                      const int32_t* a_indices,
                                      const int32_t* b_indices,
                                      const int32_t* c_d_indices,
                                      T const alpha,
                                      T const beta) {
  bool ret = GatherGemmScatterSimple<T>::Impl(
      ctx, a, b, c, d, m, n, k, a_indices, b_indices, c_d_indices, alpha, beta);

  if (!ret) {
      // we add two kernel for shape (M,N,K) = (*,32,32) here
      auto* tuner = MakeCutlassTuner<T>(
          launchKernel < float,
          cutlass_tensorop_s1688gemm_64x64_16x3_nn_align4<GatherA,
                                                          GatherB,
                                                          ScatterD>);
      tuner->AddCallBack(
          launchKernel < float,
          cutlass_tensorop_s1688f16gemm_64x64_16x10_nn_align4<GatherA,
                                                              GatherB,
                                                              ScatterD>);

      size_t key = Conv3DKey(m, n, k);

      tuner->Run(ctx,
                 key,
                 a,
                 b,
                 c,
                 d,
                 m,
                 n,
                 k,
                 a_indices,
                 b_indices,
                 c_d_indices,
                 alpha,
                 beta);
  }
}

}  // namespace sparse
}  // namespace phi
#endif
