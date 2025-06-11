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

#include "paddle/phi/kernels/index_elementwise_get_grad_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/index_elementwise.cu.h"
#include "paddle/phi/kernels/funcs/stride_utils.h"

namespace phi {

template <typename T, typename IndexT = int>
void GPUIndexElementwisePutKernel(const phi::GPUContext& ctx,
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
                           &numel);

  const int64_t* template_stride = strides_array[2];
  PADDLE_ENFORCE_NOT_NULL(template_stride,
                          ::common::errors::InvalidArgument(
                              "strides_array[2] should not be nullptr in "
                              "GPUIndexElementwiseGetKernel"));

  size_t stride_size = desired_shape.size();
  std::vector<std::vector<int64_t>> strides_vector;
  strides_vector.reserve(num_indices + 2);

  for (int i = 0; i < 2; ++i) {
    if (i < strides_array.size() && strides_array[i] != nullptr) {
      strides_vector.emplace_back(strides_array[i],
                                  strides_array[i] + stride_size);
    } else {
      strides_vector.emplace_back(stride_size, 0);
    }
  }

  std::vector<int64_t> template_vec(template_stride,
                                    template_stride + stride_size);
  for (size_t i = 0; i < num_indices; ++i) {
    strides_vector.push_back(template_vec);
  }

  auto offset_calc = funcs::make_offset_calculator<3>(
      desired_shape.size(), desired_shape.data(), strides_vector);

  const int64_t N = numel;
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
  auto stream = ctx.stream();

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
          PADDLE_ENFORCE(-sizes[i] <= index,
                         "index is less than the lower bound");
          PADDLE_ENFORCE(index < sizes[i],
                         "index is greater than or equal to the upper bound");
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
void IndexElementwiseGetGradKernel(const Context& ctx,
                                   const DenseTensor& x,
                                   const std::vector<const DenseTensor*>& index,
                                   const DenseTensor& out_grad,
                                   const std::vector<int64_t>& input_dims,
                                   const std::vector<int64_t>& input_strides,
                                   const std::vector<int64_t>& index_dims,
                                   const std::vector<int64_t>& index_strides,
                                   DenseTensor* x_grad) {
  ctx.template Alloc<T>(x_grad);
  auto dxt = phi::EigenVector<T>::Flatten(*x_grad);
  auto& place = *ctx.eigen_device();
  dxt.device(place) = dxt.constant(static_cast<T>(0));
  if (out_grad.numel() == 0) return;

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

  if (index_type == phi::DataType::INT32) {
    GPUIndexElementwisePutKernel<T, int>(ctx,
                                         x,
                                         out_grad,
                                         index,
                                         input_dims,
                                         input_strides,
                                         index_dims,
                                         index_strides,
                                         x_grad);
  } else if (index_type == phi::DataType::INT64) {
    GPUIndexElementwisePutKernel<T, int64_t>(ctx,
                                             x,
                                             out_grad,
                                             index,
                                             input_dims,
                                             input_strides,
                                             index_dims,
                                             index_strides,
                                             x_grad);
  }
}

}  // namespace phi
PD_REGISTER_KERNEL(index_elementwise_get_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::IndexElementwiseGetGradKernel,
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
