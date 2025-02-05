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

#include <climits>

#include "paddle/phi/kernels/unique_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/utils/data_type.h"
#include "paddle/phi/kernels/funcs/unique_functor.h"

namespace phi {

template <typename T, typename Context>
void UniqueKernel(const Context& context,
                  const DenseTensor& x,
                  bool return_index,
                  bool return_inverse,
                  bool return_counts,
                  const std::vector<int>& axis,
                  DataType dtype,
                  DenseTensor* out,
                  DenseTensor* indices,
                  DenseTensor* index,
                  DenseTensor* counts) {
  bool is_sorted = true;
  UniqueRawKernel<T, Context>(context,
                              x,
                              return_index,
                              return_inverse,
                              return_counts,
                              axis,
                              dtype,
                              is_sorted,
                              out,
                              indices,
                              index,
                              counts);
}

template <typename T, typename Context>
void UniqueRawKernel(const Context& context,
                     const DenseTensor& x,
                     bool return_index,
                     bool return_inverse,
                     bool return_counts,
                     const std::vector<int>& axis,
                     DataType dtype,
                     bool is_sorted,
                     DenseTensor* out,
                     DenseTensor* indices,
                     DenseTensor* index,
                     DenseTensor* counts) {
  if (dtype == phi::DataType::INT32) {
    PADDLE_ENFORCE_LE(
        x.numel(),
        INT_MAX,
        common::errors::InvalidArgument(
            "The number of elements in Input(X) should be less than or "
            "equal to INT_MAX, but received num is %d. Please set `dtype` to "
            "int64.",
            x.numel()));
  }
  if (!is_sorted) {
    phi::VisitDataType(
        dtype,
        phi::funcs::UniqueOpFunctor<Context, T>(context, out, index, &x));
    return;
  }

  if (axis.empty()) {
    phi::VisitDataTypeTiny(
        dtype,
        phi::funcs::UniqueFlattenedTensorFunctor<Context, T>(context,
                                                             x,
                                                             out,
                                                             indices,
                                                             index,
                                                             counts,
                                                             return_index,
                                                             return_inverse,
                                                             return_counts));
  } else {
    int axis_value = axis[0];
    axis_value = (axis_value == -1) ? (x.dims().size() - 1) : axis_value;
    phi::VisitDataTypeTiny(
        dtype,
        phi::funcs::UniqueDimFunctor<Context, T>(context,
                                                 x,
                                                 out,
                                                 indices,
                                                 index,
                                                 counts,
                                                 axis_value,
                                                 return_index,
                                                 return_inverse,
                                                 return_counts));
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(unique,
                   CPU,
                   ALL_LAYOUT,
                   phi::UniqueKernel,
                   float,
                   double,
                   int32_t,
                   int64_t) {
  kernel->OutputAt(1).SetDataType(phi::DataType::UNDEFINED);
  kernel->OutputAt(2).SetDataType(phi::DataType::UNDEFINED);
  kernel->OutputAt(3).SetDataType(phi::DataType::UNDEFINED);
}

PD_REGISTER_KERNEL(unique_raw,
                   CPU,
                   ALL_LAYOUT,
                   phi::UniqueRawKernel,
                   float,
                   double,
                   int32_t,
                   int64_t) {
  kernel->OutputAt(1).SetDataType(phi::DataType::UNDEFINED);
  kernel->OutputAt(2).SetDataType(phi::DataType::UNDEFINED);
  kernel->OutputAt(3).SetDataType(phi::DataType::UNDEFINED);
}
