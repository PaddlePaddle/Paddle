// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"

namespace phi {

using float16 = phi::dtype::float16;
using bfloat16 = phi::dtype::bfloat16;

template <typename T>
inline HOSTDEVICE auto copysign_func(const T& a, const T& b) {
#ifdef WIN32
  using U = typename std::conditional_t<std::is_integral<T>::value, float, T>;
  return static_cast<T>(std::copysign(static_cast<U>(a), static_cast<U>(b)));
#else
  return static_cast<T>(std::copysign(a, b));
#endif
}

inline HOSTDEVICE phi::dtype::float16 copysign_func(phi::dtype::float16 a,
                                                    phi::dtype::float16 b) {
  return phi::dtype::raw_uint16_to_float16((a.x & 0x7fff) | (b.x & 0x8000));
}

inline HOSTDEVICE phi::dtype::bfloat16 copysign_func(phi::dtype::bfloat16 a,
                                                     phi::dtype::bfloat16 b) {
  return phi::dtype::raw_uint16_to_bfloat16((a.x & 0x7fff) | (b.x & 0x8000));
}

template <typename T, typename Context>
void CopySignGradKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const DenseTensor& y,
                        const DenseTensor& out_grad,
                        DenseTensor* x_grad,
                        DenseTensor* y_grad);
}  // namespace  phi
