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
};

template <typename T, typename ReturnType, typename... Args>
static autotune::AutoTuneBase<T,
                              autotune::KernelCallback<T, ReturnType, Args...>>*
MakeCutlassTuner(ReturnType (*func)(Args...)) {
  return CutlassAutoTuner<T, ReturnType, Args...>::Instance(func);
}

}  // namespace sparse
}  // namespace phi
#endif
