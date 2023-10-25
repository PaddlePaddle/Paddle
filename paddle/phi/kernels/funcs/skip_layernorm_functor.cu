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

#include "phi/kernels/funcs/skip_layernorm_functor.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/device_context.h"

namespace phi {
namespace funcs {

template <typename T>
void SkipLayerNormFunctor<T>::operator()(const int num,
                                         const int hidden,
                                         const T *input1,
                                         const T *input2,
                                         const T *scale,
                                         const T *bias,
                                         T *output,
                                         float eps,
                                         gpuStream_t stream) {
  int block = num / hidden;
  if (hidden <= WARP_SIZE) {
    const int threads = WARP_SIZE;
    SkipLayerNormSmallKernel<T, threads><<<block, threads, 0, stream>>>(
        num, hidden, input1, input2, output, scale, bias, eps);
  } else if (hidden <= 128) {
    const int threads = 128;
    SkipLayerNormSmallKernel<T, threads><<<block, threads, 0, stream>>>(
        num, hidden, input1, input2, output, scale, bias, eps);
  } else if (hidden == 384) {
    const int threads = 384;
    SkipLayerNormSmallKernel<T, threads><<<block, threads, 0, stream>>>(
        num, hidden, input1, input2, output, scale, bias, eps);
  } else {
    const int threads = 256;
    if (hidden % 2 == 0) {
      if (std::is_same<T, float>::value) {
        SkipLayerNormKernel2<float, float2, threads>
            <<<block, threads, 0, stream>>>(
                num,
                hidden / 2,
                reinterpret_cast<const float2 *>(input1),
                reinterpret_cast<const float2 *>(input2),
                reinterpret_cast<float2 *>(output),
                reinterpret_cast<const float2 *>(scale),
                reinterpret_cast<const float2 *>(bias),
                eps);
// HIP defined __HIP_NO_HALF_CONVERSIONS__ in hip.cmake
#ifndef __HIPCC__
      } else if (std::is_same<T, __half>::value) {
        SkipLayerNormKernel2<__half, __half2, threads>
            <<<block, threads, 0, stream>>>(
                num,
                hidden / 2,
                reinterpret_cast<const __half2 *>(input1),
                reinterpret_cast<const __half2 *>(input2),
                reinterpret_cast<__half2 *>(output),
                reinterpret_cast<const __half2 *>(scale),
                reinterpret_cast<const __half2 *>(bias),
                eps);
#endif
      } else {
        assert(false);
        // should not be here
      }
    } else {
      SkipLayerNormKernel<T, threads><<<block, threads, 0, stream>>>(
          num, hidden, input1, input2, output, scale, bias, eps);
    }
  }
}

template class SkipLayerNormFunctor<float>;

// device function 'operator()' is not supportted until cuda 10.0
// HIP defined __HIP_NO_HALF_CONVERSIONS__ in hip.cmake
#if defined(PADDLE_WITH_CUDA) && CUDA_VERSION >= 10000
template class SkipLayerNormFunctor<half>;
#endif

}  // namespace funcs
}  // namespace phi
