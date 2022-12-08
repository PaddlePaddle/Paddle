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

#include "paddle/phi/kernels/diagonal_grad_kernel.h"

#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/diagonal.h"

namespace phi {

using phi::PADDLE_CUDA_NUM_THREADS;

template <typename T, typename Context>
void DiagonalGradKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const DenseTensor& out_grad,
                        int offset,
                        int axis1,
                        int axis2,
                        DenseTensor* in_grad) {
  const auto* dout = &out_grad;
  const auto* dout_data = dout->data<T>();
  auto dout_dim = dout->dims().Get();
  auto dout_dim_size = dout->dims().size();

  std::vector<int64_t> res_dout = vectorize(phi::stride(dout->dims()));
  DenseTensor dout_stride_tensor;
  paddle::framework::TensorFromVector<int64_t>(
      res_dout, dev_ctx, &dout_stride_tensor);
  int64_t* dout_stride = dout_stride_tensor.data<int64_t>();

  auto* dx = in_grad;
  auto* dx_data = dev_ctx.template Alloc<T>(dx);
  auto dx_dim = dx->dims().Get();
  auto dx_dim_size = dx->dims().size();

  std::vector<int64_t> res_dx = vectorize(phi::stride(dx->dims()));
  DenseTensor dx_stride_tensor;
  paddle::framework::TensorFromVector<int64_t>(
      res_dx, dev_ctx, &dx_stride_tensor);
  int64_t* dx_stride = dx_stride_tensor.data<int64_t>();

  const int64_t offset_ = offset;
  int64_t axis1_ = axis1 < 0 ? dx_dim_size + axis1 : axis1;
  int64_t axis2_ = axis2 < 0 ? dx_dim_size + axis2 : axis2;

  int64_t numel = dx->numel();

  int threads = PADDLE_CUDA_NUM_THREADS;
  int blocks = (numel + threads - 1) / threads;

  int64_t dout_numel = out_grad.numel();
  phi::backends::gpu::GpuMemsetAsync(
      dx_data, 0, numel * sizeof(T), dev_ctx.stream());

  switch (dx_dim_size) {
    case 2:
      funcs::DiagonalCuda<T, 2, 1><<<blocks, threads>>>(dout_data,
                                                        dx_data,
                                                        offset_,
                                                        axis1_,
                                                        axis2_,
                                                        dx_stride,
                                                        dout_stride,
                                                        numel,
                                                        dout_numel,
                                                        true);
      break;
    case 3:
      funcs::DiagonalCuda<T, 3, 2><<<blocks, threads>>>(dout_data,
                                                        dx_data,
                                                        offset_,
                                                        axis1_,
                                                        axis2_,
                                                        dx_stride,
                                                        dout_stride,
                                                        numel,
                                                        dout_numel,
                                                        true);
      break;
    case 4:
      funcs::DiagonalCuda<T, 4, 3><<<blocks, threads>>>(dout_data,
                                                        dx_data,
                                                        offset_,
                                                        axis1_,
                                                        axis2_,
                                                        dx_stride,
                                                        dout_stride,
                                                        numel,
                                                        dout_numel,
                                                        true);
      break;
    case 5:
      funcs::DiagonalCuda<T, 5, 4><<<blocks, threads>>>(dout_data,
                                                        dx_data,
                                                        offset_,
                                                        axis1_,
                                                        axis2_,
                                                        dx_stride,
                                                        dout_stride,
                                                        numel,
                                                        dout_numel,
                                                        true);
      break;
    case 6:
      funcs::DiagonalCuda<T, 6, 5><<<blocks, threads>>>(dout_data,
                                                        dx_data,
                                                        offset_,
                                                        axis1_,
                                                        axis2_,
                                                        dx_stride,
                                                        dout_stride,
                                                        numel,
                                                        dout_numel,
                                                        true);
      break;
    case 7:
      funcs::DiagonalCuda<T, 7, 6><<<blocks, threads>>>(dout_data,
                                                        dx_data,
                                                        offset_,
                                                        axis1_,
                                                        axis2_,
                                                        dx_stride,
                                                        dout_stride,
                                                        numel,
                                                        dout_numel,
                                                        true);
      break;
    case 8:
      funcs::DiagonalCuda<T, 8, 7><<<blocks, threads>>>(dout_data,
                                                        dx_data,
                                                        offset_,
                                                        axis1_,
                                                        axis2_,
                                                        dx_stride,
                                                        dout_stride,
                                                        numel,
                                                        dout_numel,
                                                        true);
      break;
    case 9:
      funcs::DiagonalCuda<T, 9, 8><<<blocks, threads>>>(dout_data,
                                                        dx_data,
                                                        offset_,
                                                        axis1_,
                                                        axis2_,
                                                        dx_stride,
                                                        dout_stride,
                                                        numel,
                                                        dout_numel,
                                                        true);
      break;
    default:
      PADDLE_THROW(errors::InvalidArgument(
          "The rank of output(input@Grad) should be less than 10, but "
          "received %d.",
          dx_dim_size));
  }
}
}  // namespace phi
PD_REGISTER_KERNEL(diagonal_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::DiagonalGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   bool,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
