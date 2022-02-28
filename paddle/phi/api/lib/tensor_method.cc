/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/api/include/tensor.h"

#include "paddle/phi/api/lib/ext_compat_utils.h"
#include "paddle/phi/core/compat/convert_utils.h"
#include "paddle/phi/core/tensor_base.h"

namespace paddle {
namespace experimental {

// declare cast api
Tensor cast(const Tensor &x, DataType out_dtype);
Tensor copy_to(const Tensor &x, Backend backend, bool blocking);

Tensor Tensor::cast(DataType target_type) const {
  return experimental::cast(*this, target_type);
}

Tensor Tensor::copy_to(Backend backend, bool blocking) const {
  return experimental::copy_to(*this, backend, blocking);
}

template <typename T>
Tensor Tensor::copy_to(const PlaceType &target_place) const {
  LOG(WARNING) << "The Tensor's `copy_to` method is deprecated since version "
                  "2.3, and will be removed in version 2.4, please use "
                  "`copy_to` method without template argument instead. "
                  "reason: copying a Tensor to another device does not need "
                  "to specify the data type template argument.";
  return copy_to(ConvertExtPlaceToBackend(target_place), /*blocking=*/false);
}

template PADDLE_API Tensor
Tensor::copy_to<float>(const PlaceType &target_place) const;
template PADDLE_API Tensor
Tensor::copy_to<double>(const PlaceType &target_place) const;
template PADDLE_API Tensor
Tensor::copy_to<int64_t>(const PlaceType &target_place) const;
template PADDLE_API Tensor
Tensor::copy_to<int32_t>(const PlaceType &target_place) const;
template PADDLE_API Tensor
Tensor::copy_to<uint8_t>(const PlaceType &target_place) const;
template PADDLE_API Tensor
Tensor::copy_to<int8_t>(const PlaceType &target_place) const;
template PADDLE_API Tensor
Tensor::copy_to<int16_t>(const PlaceType &target_place) const;
template PADDLE_API Tensor
Tensor::copy_to<bool>(const PlaceType &target_place) const;
template PADDLE_API Tensor Tensor::copy_to<phi::dtype::complex<float>>(
    const PlaceType &target_place) const;
template PADDLE_API Tensor Tensor::copy_to<phi::dtype::complex<double>>(
    const PlaceType &target_place) const;
template PADDLE_API Tensor
Tensor::copy_to<phi::dtype::float16>(const PlaceType &target_place) const;

void Tensor::copy_(const Tensor &src, bool blocking) {
  if (!src.is_initialized()) {
    return;
  }
  VLOG(3) << "Deep copy Tensor from " << src.name() << " to " << name();
  if (defined()) {
    PADDLE_ENFORCE_EQ(dtype(),
                      src.dtype(),
                      platform::errors::PreconditionNotMet(
                          "Tensor %s has different data type with Tensor %s, "
                          "Tensor Copy cannot be performed!",
                          name(),
                          src.name()));
    PADDLE_ENFORCE_EQ(impl()->type_info().id(),
                      src.impl()->type_info().id(),
                      platform::errors::PreconditionNotMet(
                          "Tensor %s has different type with Tensor %s, Tensor "
                          "Copy cannot be performed!",
                          name(),
                          src.name()));
  }
  auto copy_tensor =
      src.copy_to(phi::TransToPtenBackend(src.inner_place()), blocking);
  set_impl(copy_tensor.impl());
}

}  // namespace experimental
}  // namespace paddle
