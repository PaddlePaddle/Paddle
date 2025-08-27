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

#include "paddle/phi/kernels/index_elementwise_get_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/index_elementwise.cu.h"
#include "paddle/phi/kernels/funcs/stride_utils.h"

namespace phi {
template <typename T, typename IndexT = int>
void GPUIndexElementwiseGetKernel(const phi::GPUContext& dev_ctx,
                                  const DenseTensor& input,
                                  const std::vector<const DenseTensor*> index,
                                  const std::vector<int64_t>& input_dims,
                                  const std::vector<int64_t>& input_strides,
                                  const std::vector<int64_t>& index_dims,
                                  const std::vector<int64_t>& index_stride,
                                  const int64_t slice_offset,
                                  DenseTensor* output) {
  int64_t numel = 0;
  int64_t num_indices = 0;
  std::vector<int64_t> shape_tmp;
  std::vector<int64_t> stride_tmp;
  funcs::cal_shape_stride(index_dims, &num_indices, &shape_tmp, &stride_tmp);

  auto index_ptrs = funcs::GetIndexDataPtrs<IndexT>(index);

  auto sizes = std::array<int64_t, DDim::kMaxRank>{};
  auto strides = std::array<int64_t, DDim::kMaxRank>{};

  for (int64_t i = 0; i < num_indices; i++) {
    sizes[i] = index_dims[i];
    strides[i] = index_stride[i];
  }

  std::array<int64_t*, 3> strides_array;
  std::vector<int64_t> desired_shape;
  std::array<std::vector<int64_t>, 3> strides_vec;

  funcs::IndexGetStride<3>(input_dims,
                           input_strides,
                           phi::SizeOf(input.dtype()),
                           std::vector<int64_t>(),
                           std::vector<int64_t>(),
                           phi::SizeOf(input.dtype()),
                           shape_tmp,
                           stride_tmp,
                           phi::SizeOf(index[0]->dtype()),
                           &desired_shape,
                           &strides_array,
                           &numel,
                           strides_vec);
  auto offset_calc =
      funcs::make_offset_calculator_put<3>(desired_shape, strides_array);

  const int64_t N = output->numel();
  PADDLE_ENFORCE_GE(
      N, 0, common::errors::InvalidArgument("Output numel must >= 0"));
  PADDLE_ENFORCE_LE(
      N,
      std::numeric_limits<int32_t>::max(),
      common::errors::InvalidArgument("Output numel must <= INT32_MAX"));
  constexpr int nt = 128;
  constexpr int vt = 4;
  const dim3 block(nt);
  const dim3 grid((N + block.x * vt - 1) / (block.x * vt));
  auto stream = dev_ctx.stream();

  using dtype = funcs::OpaqueType<sizeof(T)>;

  const char* in_ptr =
      reinterpret_cast<const char*>(input.data<T>()) + slice_offset;
  char* out_ptr = reinterpret_cast<char*>(output->data<T>());
  funcs::index_elementwise_with_tensor_kernel<nt, vt>
      <<<grid, block, 0, stream>>>(N, [=] __device__(int idx) {
        const auto offsets = offset_calc.get(idx);
        char* const out_data = out_ptr + offsets[0];
        const char* const in_data = in_ptr + offsets[1];

        int64_t offset = 0;
#pragma unroll
        for (int64_t i = 0; i < num_indices; i++) {
          int64_t index =
              *reinterpret_cast<int64_t*>(index_ptrs[i] + offsets[2]);
          if (index < 0) {
            index += sizes[i];
          }
          offset += index * strides[i];
        }

        *reinterpret_cast<dtype*>(out_data) =
            *reinterpret_cast<const dtype*>(in_data + offset);
      });
}

template <typename T, typename Context>
void IndexElementwiseGetKernel(const Context& dev_ctx,
                               const DenseTensor& x,
                               const std::vector<const DenseTensor*>& index,
                               const std::vector<int64_t>& input_dims,
                               const std::vector<int64_t>& input_strides,
                               const std::vector<int64_t>& index_dims,
                               const std::vector<int64_t>& index_stride,
                               const int64_t slice_offset,
                               const bool accumulate,
                               const bool is_combined,
                               DenseTensor* out) {
  const auto& index_type = index[0]->dtype();
  PADDLE_ENFORCE_EQ(index_type == phi::DataType::INT64,
                    true,
                    common::errors::InvalidArgument(
                        "Index holds the wrong type, it holds [%s], but "
                        "desires to be [%s].",
                        index_type,
                        phi::DataType::INT64));

  auto out_dims = out->dims();
  if (out_dims.size() > 0) {
    std::vector<int64_t> output_dims(input_dims);
    out->Resize(phi::make_ddim(output_dims));
  }

  dev_ctx.template Alloc<T>(out);
  if (out->numel() == 0) return;

  GPUIndexElementwiseGetKernel<T, int64_t>(dev_ctx,
                                           x,
                                           index,
                                           input_dims,
                                           input_strides,
                                           index_dims,
                                           index_stride,
                                           slice_offset,
                                           out);
}

}  // namespace phi

PD_REGISTER_KERNEL(index_elementwise_get,
                   GPU,
                   ALL_LAYOUT,
                   phi::IndexElementwiseGetKernel,
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
