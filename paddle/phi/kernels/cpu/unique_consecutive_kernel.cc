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

#include "paddle/phi/kernels/unique_consecutive_kernel.h"
#include "paddle/phi/kernels/cpu/unique_consecutive_functor.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/errors.h"
#include "paddle/phi/core/kernel_registry.h"
<<<<<<< HEAD

#include "paddle/fluid/framework/data_type.h"
=======
#include "paddle/phi/core/utils/data_type.h"
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f

namespace phi {

template <typename T, typename Context>
void UniqueConsecutiveKernel(const Context& dev_ctx,
                             const DenseTensor& x,
                             bool return_inverse,
                             bool return_counts,
                             const std::vector<int>& axis,
                             int dtype,
                             DenseTensor* out,
                             DenseTensor* index,
                             DenseTensor* counts) {
<<<<<<< HEAD
  auto data_type = static_cast<paddle::framework::proto::VarType::Type>(dtype);
  if (data_type == paddle::framework::proto::VarType::INT32) {
=======
  auto data_type = phi::TransToPhiDataType(dtype);
  if (data_type == phi::DataType::INT32) {
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
    PADDLE_ENFORCE_LE(
        x.numel(),
        INT_MAX,
        phi::errors::InvalidArgument(
            "The number of elements in Input(X) should be less than or "
            "equal to INT_MAX, but received num is %d. Please set `dtype` to "
            "int64.",
            x.numel()));
  }

  if (axis.empty()) {
<<<<<<< HEAD
    paddle::framework::VisitDataTypeTiny(
=======
    phi::VisitDataTypeTiny(
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
        data_type,
        UniqueConsecutiveFlattenedTensorFunctor<Context, T>(
            dev_ctx, x, out, return_inverse, return_counts, index, counts));
  } else {
    int valid_axis = axis[0];
<<<<<<< HEAD
    paddle::framework::VisitDataTypeTiny(
=======
    phi::VisitDataTypeTiny(
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
        data_type,
        UniqueConsecutiveDimFunctor<Context, T>(dev_ctx,
                                                x,
                                                out,
                                                valid_axis,
                                                return_inverse,
                                                return_counts,
                                                index,
                                                counts));
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(unique_consecutive,
                   CPU,
                   ALL_LAYOUT,
                   phi::UniqueConsecutiveKernel,
                   float,
                   double,
                   int32_t,
                   int64_t) {}
