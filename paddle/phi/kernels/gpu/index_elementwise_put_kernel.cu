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

#include "paddle/phi/kernels/index_elementwise_put_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/index_elementwise.cu.h"
#include "paddle/phi/kernels/funcs/stride_utils.h"

namespace phi {

template <typename T, typename IndexT = int>
void GPUIndexElementwisePutKernel(const phi::GPUContext& dev_ctx,
                                  const DenseTensor& input,
                                  const DenseTensor& value,
                                  const std::vector<const DenseTensor*>& index,
                                  const std::vector<int64_t>& input_dims,
                                  const std::vector<int64_t>& input_strides,
                                  const std::vector<int64_t>& index_dims,
                                  const std::vector<int64_t>& index_strides,
                                  DenseTensor* output) {
  int64_t numel = 0;

  auto num_indices = index_dims.size();

  auto sizes = std::array<int64_t, 25>{};
  auto strides = std::array<int64_t, 25>{};
  for (unsigned i = 0; i < num_indices; i++) {
    sizes[i] = index_dims[i];
    strides[i] = index_strides[i];
  }
  auto index_ptrs = funcs::GetIndexDataPtrs<IndexT>(index);

  std::array<int64_t*, 3> strides_array;
  std::vector<int64_t> desired_shape;
  std::array<std::vector<int64_t>, 3> strides_vec;

  funcs::IndexPutStride<3>(input_dims,
                           input_strides,
                           phi::SizeOf(input.dtype()),
                           std::vector<int64_t>(),
                           std::vector<int64_t>(),
                           phi::SizeOf(value.dtype()),
                           common::vectorize<int64_t>(index[0]->dims()),
                           common::vectorize<int64_t>(index[0]->strides()),
                           phi::SizeOf(index[0]->dtype()),
                           &desired_shape,
                           &strides_array,
                           &numel,
                           strides_vec);

  auto offset_calc =
      funcs::make_offset_calculator_put<3>(desired_shape, strides_array);

  const int64_t N = numel;
  PADDLE_ENFORCE(N >= 0 && N <= std::numeric_limits<int32_t>::max(),
                 "N >= 0 && N <= std::numeric_limits<int32_t>::max()");
  constexpr int nt = 128;
  constexpr int vt = 4;
  const dim3 block(nt);
  const dim3 grid((N + block.x * vt - 1) / (block.x * vt));
  auto stream = dev_ctx.stream();

  using dtype = funcs::OpaqueType<sizeof(T)>;

  const char* in_ptr = reinterpret_cast<const char*>(value.data<T>());
  char* out_ptr = reinterpret_cast<char*>(output->data<T>());

  funcs::index_elementwise_kernel<nt, vt>
      <<<grid, block, 0, stream>>>(N, [=] __device__(int idx) {
        const auto offsets = offset_calc.get(idx);
        char* const out_data = out_ptr + offsets[0];
        const char* const in_data = in_ptr + offsets[1];

        int64_t offset = 0;
#pragma unroll
        for (int i = 0; i < num_indices; i++) {
          int64_t index =
              *reinterpret_cast<int64_t*>(index_ptrs[i] + offsets[2]);
          PADDLE_ENFORCE(-sizes[i] <= index && index < sizes[i],
                         "index out of bounds");
          if (index < 0) {
            index += sizes[i];
          }
          offset += index * strides[i];
        }
        *reinterpret_cast<dtype*>(out_data + offset) =
            *reinterpret_cast<const dtype*>(in_data);
      });
}

template <typename T, typename Context>
void IndexElementwisePutKernel(const Context& dev_ctx,
                               const DenseTensor& x,
                               const std::vector<const DenseTensor*>& index,
                               const DenseTensor& value,
                               const std::vector<int64_t>& input_dims,
                               const std::vector<int64_t>& input_strides,
                               const std::vector<int64_t>& index_dims,
                               const std::vector<int64_t>& index_strides,
                               DenseTensor* out) {
  const auto& index_type = index[0]->dtype();
  PADDLE_ENFORCE_EQ(
      index_type == phi::DataType::INT32 || index_type == phi::DataType::INT64,
      true,
      common::errors::InvalidArgument(
          "Index holds the wrong type, it holds [%s], but "
          "desires to be [%s] or [%s].",
          index_type,
          phi::DataType::INT32,
          phi::DataType::INT64));

  if (out->numel() == 0) return;
  dev_ctx.template Alloc<T>(out);

  if (index_type == phi::DataType::INT32) {
    GPUIndexElementwisePutKernel<T, int>(dev_ctx,
                                         x,
                                         value,
                                         index,
                                         input_dims,
                                         input_strides,
                                         index_dims,
                                         index_strides,
                                         out);
  } else if (index_type == phi::DataType::INT64) {
    GPUIndexElementwisePutKernel<T, int64_t>(dev_ctx,
                                             x,
                                             value,
                                             index,
                                             input_dims,
                                             input_strides,
                                             index_dims,
                                             index_strides,
                                             out);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(index_elementwise_put,
                   GPU,
                   ALL_LAYOUT,
                   phi::IndexElementwisePutKernel,
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
                   phi::dtype::complex<double>) {}
