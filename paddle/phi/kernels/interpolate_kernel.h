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

#include "paddle/phi/core/dense_tensor.h"

namespace phi {

template <typename T, typename Context>
void BilinearInterpKernel(
    const Context& ctx,
    const DenseTensor& x,
    paddle::optional<const DenseTensor&> out_size,
    paddle::optional<const std::vector<const DenseTensor*>> size_tensor,
    paddle::optional<const DenseTensor&> scale_tensor,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    DenseTensor* output);

template <typename T, typename Context>
void NearestInterpKernel(
    const Context& ctx,
    const DenseTensor& x,
    paddle::optional<const DenseTensor&> out_size,
    paddle::optional<const std::vector<const DenseTensor*>> size_tensor,

    paddle::optional<const DenseTensor&> scale_tensor,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    DenseTensor* output);

template <typename T, typename Context>
void TrilinearInterpKernel(
    const Context& ctx,
    const DenseTensor& x,
    paddle::optional<const DenseTensor&> out_size,
    paddle::optional<const std::vector<const DenseTensor*>> size_tensor,

    paddle::optional<const DenseTensor&> scale_tensor,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    DenseTensor* output);

template <typename T, typename Context>
void LinearInterpKernel(
    const Context& ctx,
    const DenseTensor& x,
    paddle::optional<const DenseTensor&> out_size,
    paddle::optional<const std::vector<const DenseTensor*>> size_tensor,

    paddle::optional<const DenseTensor&> scale_tensor,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    DenseTensor* output);

template <typename T, typename Context>
void BicubicInterpKernel(
    const Context& ctx,
    const DenseTensor& x,
    paddle::optional<const DenseTensor&> out_size,
    paddle::optional<const std::vector<const DenseTensor*>> size_tensor,

    paddle::optional<const DenseTensor&> scale_tensor,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    DenseTensor* output);

}  // namespace phi
