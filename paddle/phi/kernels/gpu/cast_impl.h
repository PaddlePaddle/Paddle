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

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"

namespace phi {

template <typename InT, typename OutT>
struct CastFuctor {
  __device__ __forceinline__ OutT operator()(const InT x) const {
    return static_cast<OutT>(x);
  }
};

template <typename InT, typename OutT>
void CastCUDAKernelImpl(const GPUContext& dev_ctx,
                        const DenseTensor& x,
                        DenseTensor* out) {
  std::vector<const DenseTensor*> inputs;
  std::vector<DenseTensor*> outputs;
  inputs.emplace_back(&x);
  outputs.emplace_back(out);
  dev_ctx.Alloc<OutT>(out);
  phi::funcs::ElementwiseKernel<OutT>(
      dev_ctx, inputs, &outputs, CastFuctor<InT, OutT>());
}

}  // namespace phi
