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

#include "paddle/phi/kernels/prelu_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/gpu/prelu_funcs.h"

namespace phi {

template <typename T, typename Context>
void PReluKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 const DenseTensor& alpha,
                 const std::string& data_format,
                 const std::string& mode,
                 DenseTensor* out) {
  const T* x_ptr = x.data<T>();
  T* o_ptr = dev_ctx.template Alloc<T>(out);

  const T* alpha_ptr = alpha.data<T>();
  int numel = x.numel();
  auto dim = x.dims();
  auto x_rank = dim.size();

  VLOG(4) << "dim[0]:" << dim[0] << ", dim[1]:" << dim[1] << ", dim["
          << x_rank - 1 << "]:" << dim[x_rank - 1] << ", numel:" << numel;

  if (mode == "channel") {
    bool channel_last = data_format == "NHWC";
    size_t channel = channel_last ? dim[x_rank - 1] : dim[1];
    PreluChannelWiseDirectCUDAFunctor<T> prelu_channel_wise;
    prelu_channel_wise(dev_ctx.stream(),
                       x_ptr,
                       alpha_ptr,
                       o_ptr,
                       dim[0],
                       channel,
                       channel_last,
                       numel);
  } else if (mode == "element") {
    PreluElementWiseDirectCUDAFunctor<T> prelu_element_wise;
    prelu_element_wise(
        dev_ctx.stream(), x_ptr, alpha_ptr, o_ptr, dim[0], numel);
  } else {
    PreluScalarDirectCUDAFunctor<T> prelu_scalar;
    prelu_scalar(dev_ctx.stream(), x_ptr, alpha_ptr, o_ptr, numel);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(prelu,
                   GPU,
                   ALL_LAYOUT,
                   phi::PReluKernel,
                   float,
                   phi::dtype::float16,
                   double) {}
