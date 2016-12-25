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
 * \brief   Context Projection Forward.
 *
 * \param[out]  outputs           output data.
 * \param[in]   input             input data.
 * \param[in]   weight            input weight.
 * \param[in]   sequence          input data.
 * \param[in]   context_length     consecutive rows for concatenation.
 * \param[in]   begin_pad          context start position.
 * \param[in]   is_padding         whether padding 0 or not.
 *
 */
template <DeviceType Device>
void ContextProjectionForward(Tensor& output,
                              const Tensor& input,
                              const Tensor& weight,
                              const Tensor& sequence,
                              size_t context_length,
                              int context_start,
                              size_t begin_pad,
                              bool is_padding);

}  // namespace paddle
