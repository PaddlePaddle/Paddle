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

  CallbackBase() {}
  explicit CallbackBase(FuncType& func) : func_(func) {}

  template <typename... Args>
  typename std::result_of<FuncType(Args...)>::type Run(Args... args) {
    return func_(args...);
    return func_(args...);
    return func_(args...);
  }

 private:
  FuncType func_;
};

template <typename T>
static CallbackBase<T> MakeCallBack(T kernel) {
  return CallbackBase<T>(kernel);
}

/* TODO(limingshu): A basic version. Intending to design this module by
   extending functionality of CallbackBase. However, having not found one
   container to store multiple rvalue/lvalue references hinding it from
   final achievement. */
template <typename Context, typename CallBack>
struct AutoTuneBase {
 public:
  using ContentType = typename CallbackTraits<Context>::Type;

  explicit AutoTuneBase(ContentType ctx, std::vector<CallBack> kernels)
      : ctx_(ctx), kernels_(kernels) {}

  template <typename... Args>
  CallBack PickBestAlgorithm(Args... args) {
    int idx = 0;
    phi::GpuTimer timer;
    float min_time = std::numeric_limits<float>::max();

    for (int i = 0; i < kernels_.size(); ++i) {
      ctx_.Wait();
      timer.Start(0);
      kernels_[i].Run(args...);
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
  ContentType ctx_;
  std::vector<CallBack> kernels_;
};

}  // namespace phi
