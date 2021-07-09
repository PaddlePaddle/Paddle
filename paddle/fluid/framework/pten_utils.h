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

#pragma once

#include "paddle/pten/core/base_tensor.h"
#include "paddle/pten/core/convert_utils.h"

#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace framework {

template <typename TensorImplT>
std::shared_ptr<TensorImplT> MakeTensorImpl(const Tensor& tensor,
                                            const platform::Place& place,
                                            proto::VarType::Type type) {
  auto holder = tensor.Holder();
  auto meta =
      pt::TensorMeta(tensor.dims(), pt::TransToPtenBackend(place),
                     pt::TransToPtenDataType(type),
                     pt::TransToPtenLayout(tensor.layout()), tensor.offset());
  auto tensor_impl = std::make_shared<TensorImplT>(meta);
  if (holder != nullptr) {
    tensor_impl->template ShareAllocation(tensor.Holder());
  } else {
    LOG(WARNING) << "Old Tensor holder is nullptr.";
  }
  return tensor_impl;
}

template <typename TensorImplT>
void ShareTensorImpl(TensorImplT* tensor_impl, Tensor* out) {
  out->set_type(pt::TransToProtoVarType(tensor_impl->template type()));
  out->ResetHolder(tensor_impl->template MoveMemory());
}

}  // namespace framework
}  // namespace paddle
