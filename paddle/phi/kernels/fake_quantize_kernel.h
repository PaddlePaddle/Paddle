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
void FakeQuantizeRangeAbsMaxKernel(const Context& dev_ctx,
                                   const DenseTensor& x,
                                   const DenseTensor& in_scale,
                                   const paddle::optional<DenseTensor>& iter,
                                   int window_size,
                                   int bit_length,
                                   bool is_test,
                                   int round_type,
                                   DenseTensor* out,
                                   DenseTensor* out_scale,
                                   DenseTensor* out_scales);

template <typename T, typename Context>
void FakeQuantizeAbsMaxKernel(const Context& dev_ctx,
                              const DenseTensor& x,
                              int bit_length,
                              int round_type,
                              DenseTensor* out,
                              DenseTensor* out_scale);

template <typename T, typename Context>
void FakeQuantOrWithDequantMovingAverageAbsMaxKernel(
    const Context& dev_ctx,
    const DenseTensor& x,
    const DenseTensor& in_scale,
    const paddle::optional<DenseTensor>& in_accum,
    const paddle::optional<DenseTensor>& in_state,
    float moving_rate,
    int bit_length,
    bool is_test,
    int round_type,
    DenseTensor* out,
    DenseTensor* out_scale,
    DenseTensor* out_state,
    DenseTensor* out_accum);

template <typename T, typename Context>
void FakeChannelWiseQuantizeAbsMaxKernel(const Context& dev_ctx,
                                         const DenseTensor& x,
                                         int bit_length,
                                         int round_type,
                                         int quant_axis,
                                         bool is_test,
                                         DenseTensor* out,
                                         DenseTensor* out_scale);

template <typename T, typename Context>
void FakeChannelWiseQuantizeDequantizeAbsMaxKernel(const Context& dev_ctx,
                                                   const DenseTensor& x,
                                                   int bit_length,
                                                   int round_type,
                                                   int quant_axis,
                                                   DenseTensor* out,
                                                   DenseTensor* out_scale);

template <typename T, typename Context>
void FakeQuantizeDequantizeMovingAverageAbsMaxKernel(
    const Context& dev_ctx,
    const DenseTensor& x,
    const DenseTensor& in_scale,
    const paddle::optional<DenseTensor>& in_accum,
    const paddle::optional<DenseTensor>& in_state,
    float moving_rate,
    int bit_length,
    bool is_test,
    int round_type,
    DenseTensor* out,
    DenseTensor* out_scale,
    DenseTensor* out_state,
    DenseTensor* out_accum);

template <typename T, typename Context>
void FakeQuantizeDequantizeAbsMaxKernel(const Context& dev_ctx,
                                        const DenseTensor& x,
                                        int bit_length,
                                        int round_type,
                                        DenseTensor* out,
                                        DenseTensor* out_scale);

}  // namespace phi
