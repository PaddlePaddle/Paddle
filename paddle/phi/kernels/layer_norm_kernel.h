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

#include "paddle/phi/backends/gpu/gpu_decls.h"
#include "paddle/phi/core/dense_tensor.h"

namespace phi {

template <typename T, typename Context>
void LayerNormKernel(const Context& ctx,
                     const DenseTensor& x,
                     const paddle::optional<DenseTensor>& scale,
                     const paddle::optional<DenseTensor>& bias,
                     float epsilon,
                     int begin_norm_axis,
<<<<<<< HEAD
=======
                     bool is_test,
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                     DenseTensor* out,
                     DenseTensor* mean,
                     DenseTensor* variance);

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
<<<<<<< HEAD
template <typename T, typename U>
=======
template <typename T>
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
class LayerNormDirectCUDAFunctor {
 public:
  void operator()(gpuStream_t stream,
                  const T* input,
                  std::vector<int> input_shape,
<<<<<<< HEAD
                  const U* bias,
                  const U* scale,
                  T* output,
                  U* mean,
                  U* variance,
=======
                  const T* bias,
                  const T* scale,
                  T* output,
                  T* mean,
                  T* variance,
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                  int begin_norm_axis,
                  float eps);
};
#endif

}  // namespace phi
