/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
/*! \file
  \brief Functor performing linear combination with a maximum operation used by
  epilogues.
*/

#pragma once

#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/epilogue/thread/scale_type.h"
#include "cutlass/functional.h"
#include "cutlass/half.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace thread {

/////////////////////////////////////////////////////////////////////////////////////////////////

__forceinline__ __device__ float copysignf_pos(float a, float b) {
  float r;
  r = __int_as_float(__float_as_int(a) | (__float_as_int(b) & 0x80000000));
  return r;
}

__forceinline__ __device__ float tanh_opt(float x) {
#if (__CUDACC_VER_MAJOR__ < 11) || (__CUDA_ARCH__ < 750)
  const float exp_val = -1.f * fabs(2 * x);
  return copysignf_pos((1.0f - __expf(exp_val)) / (__expf(exp_val) + 1.0f), x);
#else
  return fast_tanh(x);
#endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////
template <>
struct GELU_taylor<float> {
  static const bool kIsHeavy = true;
  CUTLASS_DEVICE
  float operator()(float const& z) const {
    float k0 = static_cast<float>(0.7978845608028654);
    float k1 = static_cast<float>(0.044715);

    return static_cast<float>(
        cutlass::constants::half<float>() * z *
        (cutlass::constants::one<float>() +
         tanh_opt(k0 * z * (cutlass::constants::one<float>() + k1 * z * z))));
  }

  using Params = LinearCombinationGenericParams<float>;

  CUTLASS_DEVICE
  float operator()(float const& scalar, Params const& params_) const {
    return this->operator()(scalar);
  }
};

}  // namespace thread
}  // namespace epilogue
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
