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

#include "paddle/phi/api/lib/tensor_copy.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/enforce.h"

namespace paddle {
namespace experimental {

template <>
ScalarBase<Tensor>::ScalarBase(const Tensor& tensor_in)
    : dtype_(tensor_in.dtype()) {  // NOLINT
  PADDLE_ENFORCE_EQ(tensor_in.numel(),
                    1,
                    phi::errors::InvalidArgument(
                        "The Scalar only supports Tensor with 1 element, but "
                        "now Tensor has `%d` elements",
                        tensor_in.numel()));
  auto tensor_in_place = tensor_in.place().GetType();
  if (tensor_in_place == phi::AllocationType::GPU) {
    Tensor dst_tensor;
    copy(tensor_in, phi::CPUPlace(), true, &dst_tensor);
    GetDataFromTensor(dst_tensor);
  } else if (tensor_in_place == phi::AllocationType::CPU) {
    GetDataFromTensor(tensor_in);
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "Now, it is not supported to construct Scalar using tensor that its "
        "Place is (%s)",
        tensor_in.place()));
  }
}

}  // namespace experimental
}  // namespace paddle
