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

#include <unordered_map>
#include "glog/logging.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/kernels/autotune/gpu_timer.h"

namespace phi {

template <typename F, typename Enable = void>
struct CallbackTraits {
  using Type = F const&;
};

template <typename F>
struct CallbackTraits<
    F,
    typename std::enable_if_t<std::is_constructible<F, F const&>::value>> {
  using Type = F;
};

template <typename F>
struct CallbackBase {
 public:
  using FuncType = typename CallbackTraits<F>::Type;

  explicit CallbackBase(FuncType& func) : func_(func) {}

  template <typename... Args>
  typename std::result_of<FuncType(Args...)>::type Run(Args... args) {
    return func_(args...);
  }

 private:
  FuncType func_;
};

// TODO(limingshu): A basic version, need to be optimized later.
template <typename Context, typename CallBack, typename ReturnType>
struct AutoTuneBase {
 public:
  AutoTuneBase(Context ctx, CallBack kernels) : kernels_(kernels), ctx_(ctx) {}

  template <typename... Args>
  ReturnType PickBestAlgorithm(Args... args) {
    ctx.wait();
    float min_time = std::numeric_limits<float>::max();
    for (auto kernel : kernels) {
      phi::GpuTimer timer;
      timer.start();
      kernel.Run(args...);
      ctx.stop();
      auto time = timer.ElapsedTime();
      if (time < min_time) {
        best_kernel = kernel
      }
    }
    return best_kernel;
  }

 private:
  Context ctx;
  CallBack kernels_;
  ReturnType best_kernel;
};

}  // namespace phi
