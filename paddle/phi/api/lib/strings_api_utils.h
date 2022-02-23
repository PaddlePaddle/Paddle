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

#pragma once

#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/api/lib/utils/storage.h"
#include "paddle/phi/core/compat/convert_utils.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/meta_tensor.h"
#include "paddle/phi/core/string_tensor.h"

namespace paddle {
namespace experimental {

/* ------------------ for input ----------------------- */
inline std::shared_ptr<phi::StringTensor> TensorToStringTensor(
    const Tensor& tensor) {
  return std::dynamic_pointer_cast<phi::StringTensor>(tensor.impl());
}

/* ----------------- for infer_meta --------------------- */
inline phi::MetaTensor MakeMetaTensor(const phi::StringTensor& tensor) {
  return phi::MetaTensor(tensor);
}
/* ------------------ for output ----------------------- */
inline phi::StringTensor* SetStringsKernelOutput(Backend backend, Tensor* out) {
  if (!out->initialized()) {
    auto place = phi::TransToPtenPlace(backend);
    auto strings_tensor = std::make_shared<phi::StringTensor>(
        paddle::memory::AllocShared(place, 0), phi::StringTensorMeta());
    out->set_impl(strings_tensor);
    return strings_tensor.get();
  }
  return static_cast<phi::StringTensor*>(out->impl().get());
}
}  // namespace experimental
}  // namespace paddle
