// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
enum class AlgorithmType { kCutlass = 1 };

template <typename T, typename ReturnType, typename... Args>
class CutlassAutoTuner
    : public autotune::
          AutoTuneBase<T, autotune::KernelCallback<T, ReturnType, Args...>> {
 public:
  template <typename Context>
  void CutlassRun(const Context& ctx,
                  const size_t key,
                  const T* const a,
                  const T* const b,
                  const T* const c,
                  T* const d,
                  const int& m,
                  const int& n,
                  const int& k,
                  const int32_t* a_indices,
                  const int32_t* b_indices,
                  const int32_t* c_d_indices,
                  T alpha,
                  T beta) {
    PADDLE_ENFORCE_GT(
        kernels_.size(),
        0,
        phi::errors::InvalidArgument(
            "kernel num must be greater than 0, now is %d", kernels_.size()));
    is_init_ = true;
    auto& cache =
        autotune::AutoTuneCache::Instance().Get(CutlassAlgorithmType::kCutlass);
    if (cache.Find(key)) {
      auto best_idx = cache.Get(key);
      kernels_[best_idx].Run(ctx,
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

    } else {
      // Set alpha to 0 and beta to 1 to avoid changing the value of d when
      // picking the best kernel
      auto best_idx = PickBestKernel(
          ctx, a, b, c, d, m, n, k, a_indices, b_indices, c_d_indices, 0, 1);
      cache.Set(key, best_idx);
      kernels_[best_idx].Run(ctx,
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
};

template <typename T, typename ReturnType, typename... Args>
static autotune::AutoTuneBase<T,
                              autotune::KernelCallback<T, ReturnType, Args...>>*
MakeCutlassTuner(ReturnType (*func)(Args...)) {
  return CutlassAutoTuner<T, ReturnType, Args...>::Instance(func);
}

size_t Conv3DKey(const int m, const int n, const int k) {
  return autotune::GetKey(m, n, k);
}

template <typename T>
void GatherGemmScatter(const phi::GPUContext& ctx,
                       const T* const a,
                       const T* const b,
                       const T* const c,
                       T* const d,
                       const int& m,
                       const int& n,
                       const int& k,
                       const int32_t* a_indices,
                       const int32_t* b_indices,
                       const int32_t* c_d_indices,
                       T alpha,
                       T beta) {
  auto* tuner = MakeCutlassTuner<T>(fp16_kernels[0]);
  for (auto i = 1; i < fp16_kernels.size(); i++)
    tuner->AddCallBack(fp16_kernels[i]);

  size_t key = Conv3DKey(m, n, k);

  tuner->CutlassRun(ctx,
                    key,
                    const_cast<const T*>(a),
                    const_cast<const T*>(b),
                    const_cast<const T*>(c),
                    const_cast<T*>(d),
                    const_cast<int&>(m),
                    const_cast<int&>(n),
                    const_cast<int&>(k),
                    const_cast<const int32_t*>(a_indices),
                    const_cast<const int32_t*>(b_indices),
                    const_cast<const int32_t*>(c_d_indices),
                    alpha,
                    beta);
}

}  // namespace sparse
}  // namespace phi
#endif
