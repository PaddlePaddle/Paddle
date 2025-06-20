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

#include "glog/logging.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#include "paddle/phi/kernels/funcs/index_impl.cu.h"
#include "paddle/phi/kernels/gpu/prelu_funcs.h"

namespace phi {

template <typename T, typename Context>
void PReluKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 const DenseTensor& alpha,
                 const std::string& data_format,
                 const std::string& mode,
                 DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  if (out && out->numel() == 0) {
    return;
  }
  const T* x_ptr = x.data<T>();
  const T* alpha_ptr = alpha.data<T>();

  int64_t numel = x.numel();
  auto dim = x.dims();
  auto x_rank = dim.size();

  VLOG(4) << "dim[0]:" << dim[0] << ", dim[1]:" << dim[1] << ", dim["
          << x_rank - 1 << "]:" << dim[x_rank - 1] << ", numel:" << numel
          << ", mode:" << mode << ", format:" << data_format;

  if (mode == "channel") {
    bool channel_last = data_format == "NHWC";
    size_t channel = channel_last ? dim[x_rank - 1] : dim[1];
    if (channel_last) {
      auto func = PReluChannelLastWiseCUDAFunctor<T>(x_ptr, alpha_ptr, channel);
      phi::IndexKernel<T, PReluChannelLastWiseCUDAFunctor<T>>(
          dev_ctx, out, func);
    } else {
      size_t plane_size = numel / dim[0] / channel;
      auto func = PReluChannelFirstWiseCUDAFunctor<T>(
          x_ptr, alpha_ptr, numel, channel, plane_size);
      phi::IndexKernel<T, PReluChannelFirstWiseCUDAFunctor<T>>(
          dev_ctx, out, func);
    }
  } else if (mode == "element") {
    size_t spatial_size = numel / dim[0];
    auto func =
        PreluElementWiseDirectCUDAFunctor<T>(x_ptr, alpha_ptr, spatial_size);
    phi::IndexKernel<T, PreluElementWiseDirectCUDAFunctor<T>>(
        dev_ctx, out, func);
  } else {
    std::vector<const DenseTensor*> ins = {&x};
    std::vector<DenseTensor*> outs = {out};
    auto func = PreluScalarDirectCUDAFunctor<T>(alpha_ptr);
    phi::funcs::ElementwiseKernel<T>(dev_ctx, ins, &outs, func);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(prelu,
                   GPU,
                   ALL_LAYOUT,
                   phi::PReluKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   double) {}
