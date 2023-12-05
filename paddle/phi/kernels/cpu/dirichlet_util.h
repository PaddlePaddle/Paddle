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

#include "paddle/phi/kernels/impl/dirichlet_kernel_impl.h"

namespace phi {

template <typename T, typename UniformSamplerT, typename NormalSamplerT>
struct GammaCPUFunctor {
  GammaCPUFunctor(const T* alpha,
                  T* gamma,
                  BaseSampler<T, UniformSamplerT> uniform,
                  BaseSampler<T, NormalSamplerT> normal)
      : alpha_(alpha), gamma_(gamma), uniform_(uniform), normal_(normal) {}

  HOST void operator()(int64_t index) {
    auto sample = sample_gamma<T, T, UniformSamplerT, NormalSamplerT>(
        alpha_[index], uniform_, normal_);
    gamma_[index] = std::max(std::numeric_limits<T>::min(), sample);
  }

  const T* alpha_;
  T* gamma_;
  BaseSampler<T, UniformSamplerT> uniform_;
  BaseSampler<T, NormalSamplerT> normal_;
};

}  // namespace phi
