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

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/index_elementwise.h"
#include "paddle/phi/kernels/funcs/stride_utils.h"

namespace phi {

template <typename T, typename IndexT = int>
void CPUIndexElementwisePutKernel(const phi::CPUContext& dev_ctx,
                                  const DenseTensor& input,
                                  const DenseTensor& value,
                                  const std::vector<const DenseTensor*>& index,
                                  const std::vector<int64_t>& input_dims,
                                  const std::vector<int64_t>& input_strides,
                                  const std::vector<int64_t>& index_dims,
                                  const std::vector<int64_t>& index_strides,
                                  const int64_t slice_offset,
                                  DenseTensor* output) {
  int64_t numel = 0;
  bool is_initialized = output->initialized();
  bool is_same_place = true;
  if (is_initialized) {
    is_same_place = (input.place() == output->place());
  }
  T* output_ = dev_ctx.template Alloc<T>(output);
  if (!is_initialized || !is_same_place) {
    phi::Copy(dev_ctx, input, dev_ctx.GetPlace(), false, output);
  }
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
                           common::vectorize<int64_t>(value.dims()),
                           common::vectorize<int64_t>(value.strides()),
                           phi::SizeOf(value.dtype()),
                           common::vectorize<int64_t>(index[0]->dims()),
                           common::vectorize<int64_t>(index[0]->strides()),
                           phi::SizeOf(index[0]->dtype()),
                           &desired_shape,
                           &strides_array,
                           &numel,
                           strides_vec);
  auto offset_calc =
      funcs::CPUmake_offset_calculator_put<3>(desired_shape, strides_array);
  const int64_t N = numel;
  PADDLE_ENFORCE(N >= 0 && N <= std::numeric_limits<int32_t>::max(),
                 "N >= 0 && N <= std::numeric_limits<int32_t>::max()");
  using dtype = funcs::OpaqueType<sizeof(T)>;
  const char* in_ptr = reinterpret_cast<const char*>(value.data<T>());
  char* out_ptr = reinterpret_cast<char*>(output_);
  for (int64_t idx = 0; idx < N; idx++) {
    const auto offsets = offset_calc.cpu_get(idx);
    char* const out_data = out_ptr + offsets[0] + slice_offset;
    const char* const in_data = in_ptr + offsets[1];
    int64_t offset = 0;
    for (size_t i = 0; i < num_indices; i++) {
      int64_t index = *reinterpret_cast<int64_t*>(index_ptrs[i] + offsets[2]);
      if (index < 0) {
        index += sizes[i];
      }
      offset += index * strides[i];
    }
    *reinterpret_cast<dtype*>(out_data + offset) =
        *reinterpret_cast<const dtype*>(in_data);
  }
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
                               const int64_t slice_offset,
                               DenseTensor* out) {
  const auto& index_type = index[0]->dtype();
  PADDLE_ENFORCE_EQ(index_type == phi::DataType::INT64,
                    true,
                    common::errors::InvalidArgument(
                        "Index holds the wrong type, it holds [%s], but "
                        "desires to be [%s].",
                        index_type,
                        phi::DataType::INT64));
  if (out && out->numel() == 0) {
    dev_ctx.template Alloc<T>(out);
    return;
  }
  if (index.empty()) {
    if (!out->initialized()) {
      phi::Copy(dev_ctx, x, dev_ctx.GetPlace(), false, out);
    }
    return;
  }
  if (out->numel() == 0) return;
  CPUIndexElementwisePutKernel<T, int64_t>(dev_ctx,
                                           x,
                                           value,
                                           index,
                                           input_dims,
                                           input_strides,
                                           index_dims,
                                           index_strides,
                                           slice_offset,
                                           out);
}

}  // namespace phi

PD_REGISTER_KERNEL(index_elementwise_put,
                   CPU,
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
