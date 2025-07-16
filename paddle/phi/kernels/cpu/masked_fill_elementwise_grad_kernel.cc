// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/index_elementwise.cu.h"
#include "paddle/phi/kernels/funcs/stride_utils.h"
#include "paddle/phi/kernels/masked_fill_grad_kernel.h"

namespace phi {
template <typename T>
void GPUMaskedFillElementwiseGrad(const phi::CPUContext& dev_ctx,
                                  const DenseTensor& x,
                                  const DenseTensor& mask,
                                  const DenseTensor& out_grad,
                                  const std::vector<int64_t>& input_dims,
                                  const std::vector<int64_t>& input_strides,
                                  const int64_t slice_offset,
                                  DenseTensor* x_grad) {
  const bool* mask_data = mask.data<bool>();
  T* x_grad_data = x_grad->data<T>();
  int64_t numel = 0;
  std::array<int64_t*, 3> strides_array;
  std::vector<int64_t> desired_shape;
  std::array<std::vector<int64_t>, 3> strides_vec;
  funcs::IndexPutStride<3>(input_dims,
                           input_strides,
                           phi::SizeOf(x.dtype()),
                           std::vector<int64_t>(),
                           std::vector<int64_t>(),
                           4,
                           common::vectorize<int64_t>(mask.dims()),
                           common::vectorize<int64_t>(mask.strides()),
                           phi::SizeOf(mask.dtype()),
                           &desired_shape,
                           &strides_array,
                           &numel,
                           strides_vec);
  auto offset_calc =
      funcs::make_offset_calculator_put<3>(desired_shape, strides_array);
  const int64_t N = numel;
  constexpr int nt = 128;
  constexpr int vt = 4;
  const dim3 block(nt);
  const dim3 grid((N + block.x * vt - 1) / (block.x * vt));
  auto stream = dev_ctx.stream();

  funcs::index_elementwise_with_tensor_kernel<nt, vt>
      <<<grid, block, 0, stream>>>(N, [=] __device__(int idx) {
        char* out_ptr = reinterpret_cast<char*>(x_grad_data);
        const auto offsets = offset_calc.get(idx);
        char* const out_data = out_ptr + offsets[0] + slice_offset;
#pragma unroll
        if (mask_data[idx]) {
          *reinterpret_cast<T*>(out_data) = 0;
        }
      });
}

template <typename T, typename Context>
void MaskedFillElementwiseGradKernel(const Context& dev_ctx,
                                     const DenseTensor& x,
                                     const DenseTensor& mask,
                                     const DenseTensor& out_grad,
                                     const Scalar& value UNUSED,
                                     const std::vector<int64_t>& input_dims,
                                     const std::vector<int64_t>& input_strides,
                                     const int64_t slice_offset,
                                     DenseTensor* x_grad) {
  if (out_grad.numel() == 0 || mask.numel() == 0) {
    // x shape [2, 1, 3], mask shape [2, 0, 3], x_grad shape [2, 1, 3]
    if (x_grad) {
      phi::Full<T, Context>(
          dev_ctx, phi::IntArray(common::vectorize(x_grad->dims())), 0, x_grad);
    }
  }

  if (x_grad) {
    dev_ctx.template Alloc<T>(x_grad);
    phi::Copy(dev_ctx, out_grad, dev_ctx.GetPlace(), false, x_grad);
  }
  CPUMaskedFillElementwiseGrad<T>(dev_ctx,
                                  x,
                                  mask,
                                  out_grad,
                                  input_dims,
                                  input_strides,
                                  slice_offset,
                                  x_grad);
  return;
}
}  // namespace phi

PD_REGISTER_KERNEL(masked_fill_elementwise_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::MaskedFillElementwiseGradKernel,
                   bool,
                   float,
                   double,
                   int,
                   int8_t,
                   int64_t,
                   int16_t,
                   uint8_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {
  kernel->InputAt(1).SetDataType(phi::DataType::BOOL);
}
