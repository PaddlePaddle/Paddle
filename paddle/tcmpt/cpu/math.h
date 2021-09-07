/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/tcmpt/core/dense_tensor.h"
#include "paddle/tcmpt/core/kernel_registry.h"
#include "paddle/tcmpt/core/selected_rows_tensor.h"

#include "paddle/tcmpt/module/scale.h"
#include "paddle/tcmpt/module/sign.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/platform/device_context.h"

namespace pt {

using CPUContext = paddle::platform::CPUDeviceContext;

template <typename T>
void Sign(const CPUContext& dev_ctx, const DenseTensor& x, DenseTensor* out);

template <typename T>
void Mean(const CPUContext& dev_ctx, const DenseTensor& x, DenseTensor* out);

template <typename T>
void Scale(const CPUContext& dev_ctx,
           const DenseTensor& x,
           float scale,
           float bias,
           bool bias_after_scale,
           DenseTensor* out) {
  module::Scale<CPUContext, T>(dev_ctx, x, scale, bias, bias_after_scale, out);
}

// template <typename T>
// void ScaleSelectedRows(const CPUContext& dev_ctx,
//         const SelectedRowsTensor& x,
//         float scale,
//         float bias,
//         bool bias_after_scale,
//         SelectedRowsTensor* out);

}  // namespace pt
