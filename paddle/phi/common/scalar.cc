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

#include "paddle/phi/common/place.h"
#include "paddle/phi/core/enforce.h"

#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/platform/place.h"
namespace paddle {
namespace experimental {

// The Tensor must have one dim
template <>
ScalarBase<phi::DenseTensor>::ScalarBase(const phi::DenseTensor& tensor_in)
    : dtype_(tensor_in.dtype()) {  // NOLINT
  PADDLE_ENFORCE_EQ(tensor_in.numel(),
                    1,
                    phi::errors::InvalidArgument(
                        "The Scalar only supports Tensor with 1 element, but "
                        "now Tensor has `%d` elements",
                        tensor_in.numel()));
  auto cpu_place = phi::CPUPlace();
  if (!paddle::platform::is_same_place(tensor_in.place(), cpu_place)) {
    phi::DenseTensor tensor;
    framework::TensorCopySync(tensor_in, cpu_place, &tensor);
    GetDataFromTensor(tensor);
  } else {
    GetDataFromTensor(tensor_in);
  }
}

}  // namespace experimental
}  // namespace paddle
