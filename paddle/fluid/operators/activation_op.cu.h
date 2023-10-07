/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "paddle/fluid/operators/activation_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_impl.cu.h"
#include "paddle/fluid/platform/bfloat16.h"
#include "paddle/phi/backends/gpu/gpu_device_function.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/kernels/funcs/activation_functor.h"

namespace paddle {
namespace operators {

template <typename T>
struct CudaSoftReluFunctor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  MPType one = static_cast<MPType>(1.0f);
  float threshold;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"threshold", &threshold}};
  }

  // soft_relu(x) = log(1 + exp(max(min(x, threshold), -threshold)))
  // threshold should not be negative
  __device__ __forceinline__ T operator()(const T arg_x) const {
    MPType x = static_cast<MPType>(arg_x);
    MPType t = static_cast<MPType>(threshold);
    MPType temp_min = x < t ? x : t;
    MPType temp_max = temp_min > -t ? temp_min : -t;
    return static_cast<T>(log(one + exp(temp_max)));
  }
};

template <typename T>
struct CudaSoftReluGradFunctor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  MPType one = static_cast<MPType>(1.0f);
  float threshold;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"threshold", &threshold}};
  }

  // dx = (out > -threshold && out < threshold) ? dout * (1 - exp(-out)) : 0
  // threshold should not be negative
  __device__ __forceinline__ T operator()(const T arg_dout,
                                          const T arg_out) const {
    MPType dout = static_cast<MPType>(arg_dout);
    MPType out = static_cast<MPType>(arg_out);
    MPType t = static_cast<MPType>(threshold);
    return (out > -t && out < t) ? static_cast<T>(dout * (one - exp(-out)))
                                 : static_cast<T>(0.0f);
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() {
    return ActBwdOpFwdDeps::kDepOut;
  }
};

}  // namespace operators
}  // namespace paddle
