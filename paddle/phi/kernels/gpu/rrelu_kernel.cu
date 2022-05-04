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

#include "paddle/phi/kernels/rrelu_kernel.h"
#include "paddle/phi/kernels/gpu/rrelu_impl.cu.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void RReluKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 const float lower,
                 const float upper,
                 bool is_test,
                 bool fix_seed,
                 int seed,
                 DenseTensor* out,
                 DenseTensor* mask) {
  out->mutable_data<T>(dev_ctx.GetPlace());
  mask->mutable_data<T>(dev_ctx.GetPlace());

  paddle::operators::RReluFwGPUKernelDriver<T>(dev_ctx,
                                               is_test,
                                               lower,
                                               upper, 
                                               fix_seed,
                                               seed,
                                               x,
                                            //    seed_tensor.get_ptr(),
                                               nullptr,
                                               mask,
                                               out);

}

}  // namespace phi

PD_REGISTER_KERNEL(rrelu,
                   GPU,
                   ALL_LAYOUT,
                   phi::RReluKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
