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

#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/index_elementwise.h"
#include "paddle/phi/kernels/funcs/stride_utils.h"

namespace phi {
template <typename T, typename IndexT, typename offset_calc_t>
void IndexEleGetGradAccKernel(
    int64_t N,
    const char* in_ptr,
    char* out_ptr,
    const std::array<char*, DDim::kMaxRank> index_ptrs,
    const std::array<int64_t, phi::DDim::kMaxRank + 1> sizes,
    const std::array<int64_t, phi::DDim::kMaxRank + 1> strides,
    int num_indices,
    offset_calc_t offset_calc) {
  for (int64_t idx = 0; idx < N; idx++) {
    const auto offsets = offset_calc.cpu_get(idx);
    char* const out_data = out_ptr + offsets[0];
    const char* const in_data = in_ptr + offsets[1];
    int64_t offset = 0;
    for (int i = 0; i < num_indices; i++) {
      int64_t index = *reinterpret_cast<int64_t*>(index_ptrs[i] + offsets[2]);
      if (index < 0) index += sizes[i];
      offset += index * strides[i];
    }
    *reinterpret_cast<T*>(out_data + offset) +=
        *reinterpret_cast<const T*>(in_data);
  }
}

template <typename T, typename IndexT = int>
void CPUIndexElementwiseGetGrad(const phi::CPUContext& dev_ctx,
                                const DenseTensor& input,
                                const DenseTensor& value,
                                const std::vector<const DenseTensor*>& index,
                                const std::vector<int64_t>& input_dims,
                                const std::vector<int64_t>& input_strides,
                                const std::vector<int64_t>& index_dims,
                                const std::vector<int64_t>& index_strides,
                                const int64_t slice_offset,
                                const bool accumulate,
                                DenseTensor* output) {
  int64_t numel = 0;
  int64_t num_indices = 0;
  std::vector<int64_t> shape_tmp;
  std::vector<int64_t> stride_tmp;
  funcs::cal_shape_stride(index_dims, &num_indices, &shape_tmp, &stride_tmp);
  auto sizes = std::array<int64_t, phi::DDim::kMaxRank + 1>{};
  auto strides = std::array<int64_t, phi::DDim::kMaxRank + 1>{};
  for (int64_t i = 0; i < num_indices; i++) {
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
                           shape_tmp,
                           stride_tmp,
                           phi::SizeOf(index[0]->dtype()),
                           &desired_shape,
                           &strides_array,
                           &numel,
                           strides_vec);
  auto offset_calc =
      funcs::CPUmake_offset_calculator_put<3>(desired_shape, strides_array);
  const int64_t N = numel;
  using dtype = funcs::OpaqueType<sizeof(T)>;
  const char* in_ptr = reinterpret_cast<const char*>(value.data<T>());
  char* out_ptr = reinterpret_cast<char*>(output->data<T>()) + slice_offset;
  if (accumulate) {
    IndexEleGetGradAccKernel<T, IndexT>(N,
                                        in_ptr,
                                        out_ptr,
                                        index_ptrs,
                                        sizes,
                                        strides,
                                        num_indices,
                                        offset_calc);
  } else {
    for (int64_t idx = 0; idx < N; idx++) {
      const auto offsets = offset_calc.cpu_get(idx);
      char* const out_data = out_ptr + offsets[0];
      const char* const in_data = in_ptr + offsets[1];
      int64_t offset = 0;
      for (int64_t i = 0; i < num_indices; i++) {
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
}

template <typename T, typename Context>
void IndexElementwiseGetGradKernel(const Context& dev_ctx,
                                   const DenseTensor& x,
                                   const std::vector<const DenseTensor*>& index,
                                   const DenseTensor& out_grad,
                                   const std::vector<int64_t>& input_dims,
                                   const std::vector<int64_t>& input_strides,
                                   const std::vector<int64_t>& index_dims,
                                   const std::vector<int64_t>& index_strides,
                                   const int64_t slice_offset,
                                   const bool accumulate,
                                   const bool is_combined,
                                   DenseTensor* x_grad) {
  dev_ctx.template Alloc<T>(x_grad);
  auto dxt = phi::EigenVector<T>::Flatten(*x_grad);
  auto& place = *dev_ctx.eigen_device();
  dxt.device(place) = dxt.constant(static_cast<T>(0));
  if (out_grad.numel() == 0) return;
  const auto& index_type = index[0]->dtype();
  PADDLE_ENFORCE_EQ(index_type == phi::DataType::INT64,
                    true,
                    common::errors::InvalidArgument(
                        "Index holds the wrong type, it holds [%s], but "
                        "desires to be [%s].",
                        index_type,
                        phi::DataType::INT32,
                        phi::DataType::INT64));
  CPUIndexElementwiseGetGrad<T, int64_t>(dev_ctx,
                                         x,
                                         out_grad,
                                         index,
                                         input_dims,
                                         input_strides,
                                         index_dims,
                                         index_strides,
                                         slice_offset,
                                         accumulate,
                                         x_grad);
}

}  // namespace phi
PD_REGISTER_KERNEL(index_elementwise_get_grad,
                   CPU,
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
