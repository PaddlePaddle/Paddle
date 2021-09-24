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

#include "paddle/fluid/framework/tcmpt_utils.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/selected_rows.h"

#include "paddle/tcmpt/api/include/dev/symbols.h"

namespace paddle {
namespace framework {

/* For DenseTensor */

template <>
std::shared_ptr<pt::DenseTensor> MakeTensorImpl<pt::DenseTensor, LoDTensor>(
    const LoDTensor& tensor, pt::Backend backend, pt::DataType dtype,
    pt::DataLayout layout) {
  auto holder = tensor.Holder();
  auto tensor_impl = std::make_shared<pt::DenseTensor>(
      pt::TensorMeta(tensor.dims(), backend, dtype, layout, tensor.offset()),
      pt::TensorStatus());

  if (holder != nullptr) {
    tensor_impl->ShareAllocation(tensor.Holder());
  } else {
    VLOG(1) << "Old LoDTensor holder is nullptr.";
  }
  return tensor_impl;
}

template <>
std::shared_ptr<pt::DenseTensor> MakeTensorImpl<pt::DenseTensor, Tensor>(
    const Tensor& tensor, pt::Backend backend, pt::DataType dtype,
    pt::DataLayout layout) {
  auto holder = tensor.Holder();
  auto tensor_impl = std::make_shared<pt::DenseTensor>(
      pt::TensorMeta(tensor.dims(), backend, dtype, layout, tensor.offset()),
      pt::TensorStatus());

  if (holder != nullptr) {
    tensor_impl->ShareAllocation(tensor.Holder());
  } else {
    VLOG(1) << "Old Tensor holder is nullptr.";
  }
  return tensor_impl;
}

template <>
std::shared_ptr<pt::SelectedRowsTensor>
MakeTensorImpl<pt::SelectedRowsTensor, SelectedRows>(const SelectedRows& tensor,
                                                     pt::Backend backend,
                                                     pt::DataType dtype,
                                                     pt::DataLayout layout) {
  auto value = tensor.value();
  auto holder = value.Holder();
  auto tensor_impl = std::make_shared<pt::SelectedRowsTensor>(
      pt::TensorMeta(value.dims(), backend, dtype, layout, value.offset()),
      pt::TensorStatus(), tensor.rows(), tensor.height());

  if (holder != nullptr) {
    tensor_impl->mutable_value()->ShareAllocation(tensor.value().Holder());
  } else {
    VLOG(1) << "Old SelectedRows holder is nullptr.";
  }
  return tensor_impl;
}

template <>
std::shared_ptr<pt::DenseTensor> MakeTensorImpl<pt::DenseTensor>(
    const LoDTensor& tensor, const platform::Place& place,
    proto::VarType::Type type) {
  return MakeTensorImpl<pt::DenseTensor, LoDTensor>(
      tensor, pt::TransToPtBackend(place), pt::TransToPtDataType(type),
      pt::TransToPtLayout(tensor.layout()));
}

template <>
std::shared_ptr<pt::DenseTensor> MakeTensorImpl<pt::DenseTensor>(
    const Tensor& tensor, const platform::Place& place,
    proto::VarType::Type type) {
  return MakeTensorImpl<pt::DenseTensor, Tensor>(
      tensor, pt::TransToPtBackend(place), pt::TransToPtDataType(type),
      pt::TransToPtLayout(tensor.layout()));
}

template <>
void ShareTensorImpl<pt::DenseTensor>(pt::DenseTensor* tensor_impl,
                                      LoDTensor* out) {
  out->ResetHolderWithType(tensor_impl->allocation(),
                           pt::TransToProtoVarType(tensor_impl->type()));
}

template <>
void ShareTensorImpl<pt::DenseTensor>(pt::DenseTensor* tensor_impl,
                                      Tensor* out) {
  out->ResetHolderWithType(tensor_impl->allocation(),
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
    VLOG(1) << "Old MKLDNN Tensor holder is nullptr.";
  }

  tensor_impl->set_format(tensor.format());
  return tensor_impl;
}

template <>
void ShareTensorImpl(pt::MKLDNNDenseTensor* tensor_impl, Tensor* out) {
  out->ResetHolderWithType(tensor_impl->allocation(),
                           pt::TransToProtoVarType(tensor_impl->type()));
  out->set_format(tensor_impl->format());
}

#endif

}  // namespace framework
}  // namespace paddle
