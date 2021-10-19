// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/pten/kernels/cpu/creation.h"

#include "paddle/pten/core/kernel_registry.h"
#include "paddle/pten/kernels/common/eigen/fill.h"

namespace pten {

template <typename T>
void FillAnyLike(const CPUContext& dev_ctx,
                 const DenseTensor& x,
                 const Scalar& val,
                 DenseTensor* out) {
  eigen::fill<CPUContext, T>(dev_ctx, out, val.to<float>());
}

}  // namespace pten

PT_REGISTER_MODULE(CreationCPU);

PT_REGISTER_KERNEL("fill_any_like",
                   CPU,
                   Any,
                   pten::FillAnyLike,
                   float,
                   double,
                   int,
                   int64_t,
                   bool,
                   paddle::platform::float16) {}
