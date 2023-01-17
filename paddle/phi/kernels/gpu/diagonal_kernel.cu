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

#include "paddle/phi/kernels/diagonal_kernel.h"

#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/diagonal.h"

namespace phi {
using phi::PADDLE_CUDA_NUM_THREADS;
template <typename T, typename Context>
void DiagonalKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    int offset,
                    int axis1,
                    int axis2,
                    DenseTensor* out) {
  auto* input = &x;
  const auto* input_data = input->data<T>();
  auto input_dim = input->dims().Get();
  auto input_dim_size = input->dims().size();

  std::vector<int64_t> res_in = vectorize(phi::stride(input->dims()));
  DenseTensor input_stride_tensor;
  paddle::framework::TensorFromVector<int64_t>(
      res_in, dev_ctx, &input_stride_tensor);
  int64_t* input_stride = input_stride_tensor.data<int64_t>();

  auto* output = out;
  auto* output_data = dev_ctx.template Alloc<T>(out);
  auto output_dim = output->dims().Get();
  auto output_dim_size = output->dims().size();

  std::vector<int64_t> res_out = vectorize(phi::stride(output->dims()));
  DenseTensor output_stride_tensor;
  paddle::framework::TensorFromVector<int64_t>(
      res_out, dev_ctx, &output_stride_tensor);
  int64_t* output_stride = output_stride_tensor.data<int64_t>();

  const int64_t offset_ = offset;
  int64_t axis1_ = axis1 < 0 ? input_dim_size + axis1 : axis1;
  int64_t axis2_ = axis2 < 0 ? input_dim_size + axis2 : axis2;
  int64_t numel = input->numel();

  int threads = PADDLE_CUDA_NUM_THREADS;
  int blocks = (numel + threads - 1) / threads;

  switch (input_dim_size) {
    case 2:
      funcs::DiagonalCuda<T, 2, 1><<<blocks, threads>>>(input_data,
                                                        output_data,
                                                        offset_,
                                                        axis1_,
                                                        axis2_,
                                                        input_stride,
                                                        output_stride,
                                                        numel,
                                                        false);
      break;
    case 3:
      funcs::DiagonalCuda<T, 3, 2><<<blocks, threads>>>(input_data,
                                                        output_data,
                                                        offset_,
                                                        axis1_,
                                                        axis2_,
                                                        input_stride,
                                                        output_stride,
                                                        numel,
                                                        false);
      break;
    case 4:
      funcs::DiagonalCuda<T, 4, 3><<<blocks, threads>>>(input_data,
                                                        output_data,
                                                        offset_,
                                                        axis1_,
                                                        axis2_,
                                                        input_stride,
                                                        output_stride,
                                                        numel,
                                                        false);
      break;
    case 5:
      funcs::DiagonalCuda<T, 5, 4><<<blocks, threads>>>(input_data,
                                                        output_data,
                                                        offset_,
                                                        axis1_,
                                                        axis2_,
                                                        input_stride,
                                                        output_stride,
                                                        numel,
                                                        false);
      break;
    case 6:
      funcs::DiagonalCuda<T, 6, 5><<<blocks, threads>>>(input_data,
                                                        output_data,
                                                        offset_,
                                                        axis1_,
                                                        axis2_,
                                                        input_stride,
                                                        output_stride,
                                                        numel,
                                                        false);
      break;
    case 7:
      funcs::DiagonalCuda<T, 7, 6><<<blocks, threads>>>(input_data,
                                                        output_data,
                                                        offset_,
                                                        axis1_,
                                                        axis2_,
                                                        input_stride,
                                                        output_stride,
                                                        numel,
                                                        false);
      break;
    case 8:
      funcs::DiagonalCuda<T, 8, 7><<<blocks, threads>>>(input_data,
                                                        output_data,
                                                        offset_,
                                                        axis1_,
                                                        axis2_,
                                                        input_stride,
                                                        output_stride,
                                                        numel,
                                                        false);
      break;
    case 9:
      funcs::DiagonalCuda<T, 9, 8><<<blocks, threads>>>(input_data,
                                                        output_data,
                                                        offset_,
                                                        axis1_,
                                                        axis2_,
                                                        input_stride,
                                                        output_stride,
                                                        numel,
                                                        false);
      break;
    default:
      PADDLE_THROW(errors::InvalidArgument(
          "The rank of input should be less than 10, but received %d.",
          input_dim_size));
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(diagonal,
                   GPU,
                   ALL_LAYOUT,
                   phi::DiagonalKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   bool,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
