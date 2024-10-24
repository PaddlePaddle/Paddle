/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/common/scalar.h"

#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/tensor_utils.h"
namespace paddle::experimental {

// The Tensor must have one dim
template <>
ScalarBase<phi::DenseTensor>::ScalarBase(const phi::DenseTensor& tensor_in)
    : dtype_(tensor_in.dtype()) {  // NOLINT
  PADDLE_ENFORCE_EQ(tensor_in.numel(),
                    1,
                    common::errors::InvalidArgument(
                        "The Scalar only supports Tensor with 1 element, but "
                        "now Tensor has `%d` elements",
                        tensor_in.numel()));
  auto cpu_place = phi::CPUPlace();
  if (tensor_in.place().GetType() != phi::AllocationType::CPU) {
    phi::DenseTensor tensor;
    phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
    auto dev_ctx = pool.Get(tensor_in.place());
    phi::Copy(*dev_ctx, tensor_in, cpu_place, true, &tensor);
    GetDataFromTensor(tensor);
  } else {
    GetDataFromTensor(tensor_in);
  }
}

bool operator==(const Scalar& lhs, const Scalar& rhs) {
  return lhs.operator==(rhs);
}
bool operator!=(const Scalar& lhs, const Scalar& rhs) {
  return lhs.operator!=(rhs);
}

std::ostream& operator<<(std::ostream& os, const Scalar& s) {
  return os << s.ToString();
}
}  // namespace paddle::experimental
