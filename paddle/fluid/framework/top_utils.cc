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

#include "paddle/fluid/framework/top_utils.h"

namespace paddle {
namespace framework {

/* For DenseTensor */

template <>
std::shared_ptr<pt::DenseTensor> MakeTensorImpl<pt::DenseTensor>(
    const Tensor& tensor, pt::Backend backend, pt::DataType dtype,
    pt::DataLayout layout) {
  auto holder = tensor.Holder();
  auto tensor_impl = std::make_shared<pt::DenseTensor>(
      pt::TensorMeta(tensor.dims(), backend, dtype, layout, tensor.offset()),
      pt::TensorStatus());

  if (holder != nullptr) {
    tensor_impl->ShareAllocation(tensor.Holder());
  } else {
    LOG(WARNING) << "Old Tensor holder is nullptr.";
  }
  return tensor_impl;
}

template <>
std::shared_ptr<pt::DenseTensor> MakeTensorImpl<pt::DenseTensor>(
    const Tensor& tensor, const platform::Place& place,
    proto::VarType::Type type) {
  return MakeTensorImpl<pt::DenseTensor>(tensor, pt::TransToPtBackend(place),
                                         pt::TransToPtDataType(type),
                                         pt::TransToPtLayout(tensor.layout()));
}

template <>
void ShareTensorImpl<pt::DenseTensor>(pt::DenseTensor* tensor_impl,
                                      Tensor* out) {
  out->ResetHolderWithType(tensor_impl->MoveMemory(),
                           pt::TransToProtoVarType(tensor_impl->type()));
}

/* For MKLDNNDenseTensor (move this part into a single file later) */
#ifdef PADDLE_WITH_MKLDNN

template <>
std::shared_ptr<pt::MKLDNNDenseTensor> MakeTensorImpl<pt::MKLDNNDenseTensor>(
    const Tensor& tensor, const platform::Place& place,
    proto::VarType::Type type) {
  auto holder = tensor.Holder();
  auto tensor_impl = std::make_shared<pt::MKLDNNDenseTensor>(
      pt::TensorMeta(tensor.dims(), pt::TransToPtBackend(place),
                     pt::TransToPtDataType(type),
                     pt::TransToPtLayout(tensor.layout()), tensor.offset()),
      pt::TensorStatus());

  if (holder != nullptr) {
    tensor_impl->ShareAllocation(tensor.Holder());
  } else {
    LOG(WARNING) << "Old MKLDNN Tensor holder is nullptr.";
  }

  tensor_impl->set_format(tensor.format());
  return tensor_impl;
}

template <>
void ShareTensorImpl(pt::MKLDNNDenseTensor* tensor_impl, Tensor* out) {
  out->ResetHolderWithType(tensor_impl->MoveMemory(),
                           pt::TransToProtoVarType(tensor_impl->type()));
  out->set_format(tensor_impl->format());
}

#endif

}  // namespace framework
}  // namespace paddle
