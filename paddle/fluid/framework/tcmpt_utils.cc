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

#include "paddle/fluid/framework/variable.h"
#include "paddle/tcmpt/api/include/core.h"
#include "paddle/tcmpt/api/include/symbols.h"

namespace paddle {
namespace framework {

// TODO(chenweihang, shixiaowei): adapt SelectedRows

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

std::shared_ptr<pt::TensorInterface> InputVariableToPtTensor(
    const framework::Variable& variable, const pt::TensorArgDef& arg_def) {
  auto expected_place = pt::TransToFluidPlace(arg_def.backend);

  if (variable.template IsType<framework::LoDTensor>()) {
    const auto& tensor = variable.template Get<framework::LoDTensor>();
    if (!platform::is_same_place(tensor.place(), expected_place)) {
      framework::LoDTensor tmp_tensor;
      framework::TensorCopySync(tensor, expected_place, &tmp_tensor);
      auto pt_in =
          framework::MakeTensorImpl<pt::DenseTensor, framework::LoDTensor>(
              tmp_tensor, arg_def.backend, arg_def.dtype, arg_def.layout);
      return pt_in;
    } else {
      auto pt_in =
          framework::MakeTensorImpl<pt::DenseTensor, framework::LoDTensor>(
              tensor, arg_def.backend, arg_def.dtype, arg_def.layout);
      return pt_in;
    }
  } else if (variable.template IsType<framework::SelectedRows>()) {
    // TODO(chenweihang): now we don't deal with row and height
    // by xiaowei's advice
    const auto& tensor = variable.template Get<framework::SelectedRows>();
    if (!platform::is_same_place(tensor.value().place(), expected_place)) {
      framework::Tensor tmp_tensor;
      TensorCopySync(tensor.value(), expected_place, &tmp_tensor);
      // TODO(chenweihang): adapt SelectedRows by xiaowei's design
      auto pt_in =
          framework::MakeTensorImpl<pt::DenseTensor, framework::Tensor>(
              tmp_tensor, arg_def.backend, arg_def.dtype, arg_def.layout);
      return pt_in;
    } else {
      auto pt_in =
          framework::MakeTensorImpl<pt::DenseTensor, framework::Tensor>(
              tensor.value(), arg_def.backend, arg_def.dtype, arg_def.layout);
      return pt_in;
    }
  } else {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Unsupported shared input `%s` type now when call pt kernel.",
        framework::ToTypeName(variable.Type())));
  }
  return nullptr;
}

std::shared_ptr<pt::TensorInterface> OutputVariableToPtTensor(
    framework::Variable* variable, const pt::TensorArgDef& arg_def) {
  // mutable_data before run kernel, to avoid share output form
  // KernelContext to original tensor
  if (variable->template IsType<framework::LoDTensor>()) {
    auto* tensor = variable->template GetMutable<framework::LoDTensor>();
    tensor->mutable_data(pt::TransToFluidPlace(arg_def.backend),
                         pt::TransToProtoVarType(arg_def.dtype));
    auto pt_out =
        framework::MakeTensorImpl<pt::DenseTensor, framework::LoDTensor>(
            *tensor, arg_def.backend, arg_def.dtype, arg_def.layout);
    return pt_out;
  } else if (variable->template IsType<framework::SelectedRows>()) {
    auto* tensor = variable->template GetMutable<framework::SelectedRows>();
    tensor->mutable_value()->mutable_data(
        pt::TransToFluidPlace(arg_def.backend),
        pt::TransToProtoVarType(arg_def.dtype));
    // TODO(chenweihang): adapt SelectedRows by xiaowei's design,
    // here the row and height will lost in output!
    auto pt_out = framework::MakeTensorImpl<pt::DenseTensor, framework::Tensor>(
        tensor->value(), arg_def.backend, arg_def.dtype, arg_def.layout);
    return pt_out;
  } else {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Unsupported shared output `%s` type now when call pt kernel.",
        framework::ToTypeName(variable->Type())));
  }

  return nullptr;
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
