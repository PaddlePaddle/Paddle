/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/math/blas.h"

namespace paddle {
namespace operators {
namespace math {

template <typename DeviceContext, typename T>
struct MultiHeadGPUCompute {
  static void compute(const DeviceContext &dev_ctx, int head_num,
                      const framework::DDim &mat_q,
                      const framework::DDim &mat_k,
                      const framework::DDim &mat_v, const T *Q, const T *K,
                      const T *V, const T *bias_q, const T *bias_k,
                      const T *bias_v, const T *bias_qk, T *out, T alpha,
                      T beta);
};

}  // namespace math
}  // namespace operators
}  // namespace paddle
