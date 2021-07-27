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

#include "paddle/pten/core/selected_rows.h"

// In fact, it is ugly to use such a complicated include
// relationship when coding.
// After the kernel registration module is completed, the calculation
// function should be reused by calling the kernel in global KernelMap.
#include "paddle/pten/cpu/math.h"
#include "paddle/pten/cuda/math.h"
#include "paddle/pten/npu/math.h"
#include "paddle/pten/xpu/math.h"

// See Note [ Why still include the fluid headers? ]

namespace pt {

template <typename T>
void Scale(const CPUDeviceContext& dev_ctx,
           const SelectedRowsTensor& x,
           float scale,
           float bias,
           bool bias_after_scale,
           SelectedRowsTensor* out) {
  out->set_rows(x.rows());
  out->set_height(x.height());
  Scale<T>(dev_ctx, x.value(), scale, bias, bias_after_scale, out->value());
}

}  // namespace pt
