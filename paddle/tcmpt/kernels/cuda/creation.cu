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

#include "paddle/tcmpt/kernels/cuda/creation.h"

#include "paddle/tcmpt/core/kernel_registry.h"
#include "paddle/tcmpt/kernels/common/eigen/fill.h"

namespace pt {

template <typename T>
void FillAnyLike(const CUDAContext& dev_ctx,
                 const DenseTensor& x,
                 const Scalar& val,
                 DenseTensor* out) {
  eigen::fill<CUDAContext, T>(dev_ctx, out, val.to<float>());
}

}  // namespace pt

PT_REGISTER_MODULE(CreationCUDA);

PT_REGISTER_KERNEL("fill_any_like",
                   CUDA,
                   Any,
                   pt::FillAnyLike,
                   float,
                   double,
                   int,
                   int64_t,
                   bool,
                   paddle::platform::float16) {}
