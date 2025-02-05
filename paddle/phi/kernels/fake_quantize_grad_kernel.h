// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

namespace phi {

template <typename T, typename Context>
void FakeChannelWiseQuantizeDequantizeAbsMaxGradKernel(const Context& dev_ctx,
                                                       const DenseTensor& dout,
                                                       int bit_length,
                                                       int round_type,
                                                       int quant_axis,
                                                       DenseTensor* dx);

template <typename T, typename Context>
void FakeQuantizeDequantizeAbsMaxGradKernel(const Context& dev_ctx,
                                            const DenseTensor& dout,
                                            int bit_length,
                                            int round_type,
                                            DenseTensor* dx);

template <typename T, typename Context>
void FakeQuantizeDequantizeMovingAverageAbsMaxGradKernel(
    const Context& dev_ctx,
    const DenseTensor& dout,
    float moving_rate,
    int bit_length,
    bool is_test,
    int round_type,
    DenseTensor* dx);

}  // namespace phi
