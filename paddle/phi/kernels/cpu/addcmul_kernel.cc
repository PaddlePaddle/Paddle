// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/addcmul_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/addcmul_functor.h"
#include "paddle/phi/kernels/funcs/common_shape.h"
#include "paddle/phi/kernels/funcs/elementwise_utils.h"

namespace phi {

template <typename T, typename Context>
void AddcmulKernel(const Context& dev_ctx,
                   const DenseTensor& input,
                   const DenseTensor& tensor1,
                   const DenseTensor& tensor2,
                   const Scalar& value,
                   DenseTensor* out) {
  T* out_data = dev_ctx.template Alloc<T>(out);
  funcs::AddcmulFunctor<T> functor(funcs::GetAddcmulValue<T>(value));
  const int64_t numel = out->numel();
  if (numel == 0) {
    return;
  }

  const T* input_data = input.data<T>();
  const T* tensor1_data = tensor1.data<T>();
  const T* tensor2_data = tensor2.data<T>();

  const DDim& out_dims = out->dims();
  if (input.dims() == out_dims && tensor1.dims() == out_dims &&
      tensor2.dims() == out_dims) {
    for (int64_t i = 0; i < numel; ++i) {
      out_data[i] = functor(input_data[i], tensor1_data[i], tensor2_data[i]);
    }
    return;
  }

  // NOTE(large-tensor): tensor rank is a small integer
  const int rank = out_dims.size();
  auto out_dims_array = common::vectorize<int64_t>(out_dims);
  auto input_dims_array =
      common::vectorize<int64_t>(funcs::ExtendDims2Rank(input.dims(), rank));
  auto tensor1_dims_array =
      common::vectorize<int64_t>(funcs::ExtendDims2Rank(tensor1.dims(), rank));
  auto tensor2_dims_array =
      common::vectorize<int64_t>(funcs::ExtendDims2Rank(tensor2.dims(), rank));
  std::vector<int64_t> index_array(rank, 0);
  for (int64_t i = 0; i < numel; ++i) {
    const int64_t input_index = funcs::GetElementwiseIndex<int64_t>(
        input_dims_array.data(), rank, index_array.data());
    const int64_t tensor1_index = funcs::GetElementwiseIndex<int64_t>(
        tensor1_dims_array.data(), rank, index_array.data());
    const int64_t tensor2_index = funcs::GetElementwiseIndex<int64_t>(
        tensor2_dims_array.data(), rank, index_array.data());
    out_data[i] = functor(input_data[input_index],
                          tensor1_data[tensor1_index],
                          tensor2_data[tensor2_index]);
    funcs::UpdateElementwiseIndexArray<int64_t>(
        out_dims_array.data(), rank, index_array.data());
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(addcmul,
                   CPU,
                   ALL_LAYOUT,
                   phi::AddcmulKernel,
                   float,
                   double,
                   phi::float16,
                   phi::bfloat16,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t,
                   phi::complex64,
                   phi::complex128) {}
