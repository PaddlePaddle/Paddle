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
#include "paddle/phi/kernels/funcs/index_elementwise.cu.h"
#include "paddle/phi/kernels/funcs/stride_utils.h"
#include "paddle/phi/kernels/masked_fill_kernel.h"

namespace phi {

template <typename T>
void CPUMaskedFillElementwise(const phi::CPUContext& dev_ctx,
                              const DenseTensor& x,
                              const DenseTensor& mask,
                              const Scalar& value,
                              const std::vector<int64_t>& input_dims,
                              const std::vector<int64_t>& input_strides,
                              const int64_t slice_offset,
                              DenseTensor* output) {
  const bool* mask_data = mask.data<bool>();
  bool is_initialized = output->initialized();
  bool is_same_place = true;
  if (is_initialized) {
    is_same_place = (x.place() == output->place());
  }
  dev_ctx.template Alloc<T>(output);
  T* output_data = output->data<T>();
  const T value_data = value.to<T>();
  if (!is_initialized || !is_same_place) {
    phi::Copy(dev_ctx, x, dev_ctx.GetPlace(), false, output);
  }
  int64_t numel = 0;
  std::array<int64_t*, 3> strides_array;
  std::vector<int64_t> desired_shape;
  std::array<std::vector<int64_t>, 3> strides_vec;
  funcs::IndexPutStride<3>(input_dims,
                           input_strides,
                           phi::SizeOf(x.dtype()),
                           std::vector<int64_t>(),
                           std::vector<int64_t>(),
                           phi::SizeOf(value.dtype()),
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
        char* out_ptr = reinterpret_cast<char*>(output_data);
        const auto offsets = offset_calc.get(idx);
        char* const out_data = out_ptr + offsets[0] + slice_offset;
#pragma unroll
        if (mask_data[idx]) {
          *reinterpret_cast<T*>(out_data) = value_data;
        }
      });
}

template <typename T, typename Context>
void MaskedFillElementwiseKernel(const Context& dev_ctx,
                                 const DenseTensor& x,
                                 const DenseTensor& mask,
                                 const Scalar& value,
                                 const std::vector<int64_t>& input_dims,
                                 const std::vector<int64_t>& input_strides,
                                 const int64_t slice_offset,
                                 DenseTensor* out) {
  if (x.numel() == 0 || mask.numel() == 0) {
    dev_ctx.template Alloc<T>(out);
    return;
  }

  CPUMaskedFillElementwise<T>(
      dev_ctx, x, mask, value, input_dims, input_strides, slice_offset, out);
  return;
}

}  // namespace phi

PD_REGISTER_KERNEL(masked_fill_elementwise,
                   CPU,
                   ALL_LAYOUT,
                   phi::MaskedFillElementwiseKernel,
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
