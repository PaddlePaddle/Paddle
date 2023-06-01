/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/phi/api/ext/exception.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/dist_tensor.h"

namespace paddle {
namespace experimental {
class DistTensorHijackHelper {
 public:
  static Tensor UnWrap(const Tensor& tensor) {
    if (!tensor.is_dist_tensor()) {
      return tensor;
    }
    Tensor unwrapped;
    auto impl = std::dynamic_pointer_cast<phi::DistTensor>(tensor.impl());
    if (impl) {
      unwrapped.set_impl(impl->local_tensor());
      unwrapped.set_name("unwrapped tensor for " + tensor.name());
    }
    return unwrapped;
  }

  static Tensor Wrap(const Tensor& tensor) {
    auto impl = std::dynamic_pointer_cast<phi::DenseTensor>(tensor.impl());
    std::shared_ptr<phi::DistTensor> dist_tensor =
        std::make_shared<phi::DistTensor>(impl);
    Tensor wrapped;
    wrapped.set_impl(std::move(dist_tensor));
    return wrapped;
  }
};
}  // namespace experimental
}  // namespace paddle
