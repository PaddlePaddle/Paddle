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

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/kernels/funcs/for_range.h"

// TODO(paddle-dev): Remove this file when we can call related Kernel directly

namespace phi {
namespace funcs {

template <typename T>
struct PowFunctor {
  PowFunctor(const T* input, T* output, int64_t numel, T exp)
      : input_(input), output_(output), numel_(numel), exp_(exp) {}

  HOSTDEVICE void operator()(int64_t idx) const {
    output_[idx] = pow(input_[idx], exp_);
  }
  const T* input_;
  T* output_;
  int64_t numel_;
  T exp_;
};

template <typename T, typename Context>
DenseTensor Pow(const Context& dev_ctx, const DenseTensor& x, T exp) {
  DenseTensor out;
  out.Resize(x.dims());
  dev_ctx.template Alloc<T>(&out);
  auto for_range = ForRange<Context>(dev_ctx, x.numel());
  PowFunctor<T> functor(x.data<T>(), out.data<T>(), x.numel(), exp);
  for_range(functor);
  return out;
}

}  // namespace funcs
}  // namespace phi
