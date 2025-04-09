/* Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include "paddle/phi/kernels/custom/c_allreduce_kernel_impl.h"
#ifdef PADDLE_WITH_CUSTOM_DEVICE
namespace phi {
template <typename T, typename Context>
void CAllReduceProdKernel(const Context& dev_ctx,
                          const DenseTensor& x_in,
                          int ring_id,
                          bool use_calc_stream,
                          bool use_model_parallel,
                          DenseTensor* out) {
  CAllReduceKernel<T, Context, phi::ccl::CCLReduceOp::PRODUCT>(
      dev_ctx, x_in, ring_id, use_calc_stream, use_model_parallel, out);
}
}  // namespace phi

PD_REGISTER_KERNEL(c_allreduce_prod,
                   Custom,
                   ALL_LAYOUT,
                   phi::CAllReduceProdKernel,
                   float,
                   double,
                   int32_t,
                   int64_t,
                   phi::dtype::float16) {}
#endif
