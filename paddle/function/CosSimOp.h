/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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

#include "Function.h"

namespace paddle {

/**
 * \brief   Cosine Similarity Forward.
 * for each row i,
 * out[i] = scale * cos(in1[i], in2[i])
 *        = scale * \sum_j (in1[i][j] * in2[i][j]) /
 *                  sqrt(sum_j (in1[i][j]^2) * sum_j (in2[i][j])^2)
 *
 * \param[out]  output            output data.
 * \param[in]   intput1           input data.
 * \param[in]   intput2           input data.
 * \param[in]   scale             default 1.0.
 *
 */
template <DeviceType Device>
void CosSimForward(typename MatrixT<Device>::type* output,
                   const typename MatrixT<Device>::type* input1,
                   const typename MatrixT<Device>::type* input2,
                   real scale);

}  // namespace paddle
