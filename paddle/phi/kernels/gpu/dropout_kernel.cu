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

#include "paddle/fluid/operators/dropout_impl.cu.h"
#include "paddle/phi/kernels/dropout_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void DropoutRawKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      paddle::optional<const DenseTensor&> seed_tensor,
                      float p,
                      bool is_test,
                      const std::string& mode,
                      int seed,
                      bool fix_seed,
                      DenseTensor* out,
                      DenseTensor* mask) {
  out->mutable_data<T>(dev_ctx.GetPlace());
  float dropout_prob = p;
  bool upscale_in_train = (mode == "upscale_in_train");
  mask->mutable_data<uint8_t>(dev_ctx.GetPlace());

  paddle::operators::DropoutFwGPUKernelDriver<T>(dev_ctx,
                                                 is_test,
                                                 mode,
                                                 dropout_prob,
                                                 upscale_in_train,
                                                 fix_seed,
                                                 seed,
                                                 x,
                                                 seed_tensor.get_ptr(),
                                                 mask,
                                                 out);
}

}  // namespace phi

PD_REGISTER_KERNEL(dropout,
                   GPU,
                   ALL_LAYOUT,
                   phi::DropoutRawKernel,
                   float,
                   double,
                   phi::dtype::bfloat16,
                   phi::dtype::float16) {}
