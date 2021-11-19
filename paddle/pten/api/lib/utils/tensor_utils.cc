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

#include "paddle/pten/api/lib/utils/tensor_utils.h"

#include <vector>

#include "paddle/pten/core/compat_utils.h"

namespace paddle {
namespace experimental {

template <typename DstLoD, typename SrcLoD>
void SetLoD(DstLoD* dst, const SrcLoD& src) {
  dst->reserve(src.size());
  dst->clear();
  for (auto&& v : src) {
    dst->emplace_back(v);
  }
}

std::unique_ptr<pten::DenseTensor> MakePtenDenseTensor(
    const paddle::framework::Tensor& src) {
  pten::DenseTensorMeta meta{pten::TransToPtenDataType(src.type()),
                             src.dims(),
                             pten::TransToPtenDataLayout(src.layout())};
  auto shared_storage =
      pten::make_intrusive<SharedStorage>(src.Holder(), src.offset());
  return std::make_unique<pten::DenseTensor>(std::move(shared_storage),
                                             std::move(meta));
}

std::unique_ptr<pten::DenseTensor> MakePtenDenseTensor(
    const paddle::framework::LoDTensor& src) {
  pten::DenseTensorMeta meta{pten::TransToPtenDataType(src.type()),
                             src.dims(),
                             pten::TransToPtenDataLayout(src.layout())};
  SetLoD(&meta.lod, src.lod());
  auto shared_storage =
      pten::make_intrusive<SharedStorage>(src.Holder(), src.offset());

  return std::make_unique<pten::DenseTensor>(std::move(shared_storage),
                                             std::move(meta));
}

std::unique_ptr<pten::TensorBase> MakePtenTensorBaseFromVar(
    const framework::Variable& variable, const pten::TensorArgDef& arg_def) {
  auto expected_place = pten::TransToFluidPlace(arg_def.backend);

  if (variable.IsType<framework::LoDTensor>()) {
    const auto& tensor = variable.Get<framework::LoDTensor>();
    if (!platform::is_same_place(tensor.place(), expected_place)) {
      framework::LoDTensor tmp_tensor;
      framework::TensorCopySync(tensor, expected_place, &tmp_tensor);
      return MakePtenDenseTensor(tmp_tensor);
    } else {
      return MakePtenDenseTensor(tensor);
    }
  } else if (variable.IsType<framework::SelectedRows>()) {
    // TODO(chenweihang): now we don't deal with row and height
    // by xiaowei's advice
    const auto& tensor = variable.Get<framework::SelectedRows>();
    if (!platform::is_same_place(tensor.value().place(), expected_place)) {
      framework::Tensor tmp_tensor;
      TensorCopySync(tensor.value(), expected_place, &tmp_tensor);
      // TODO(chenweihang): adapt SelectedRows by xiaowei's design
      return MakePtenDenseTensor(tmp_tensor);
    } else {
      return MakePtenDenseTensor(tensor.value());
    }
  } else {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Unsupported shared input `%s` type now when call pt kernel.",
        framework::ToTypeName(variable.Type())));
  }
  return {};
}

std::unique_ptr<pten::TensorBase> MakePtenTensorBaseFromVar(
    framework::Variable* variable, const pten::TensorArgDef& arg_def) {
  // mutable_data before run kernel, to avoid share output form
  // KernelContext to original tensor
  if (variable->template IsType<framework::LoDTensor>()) {
    auto* tensor = variable->template GetMutable<framework::LoDTensor>();
    tensor->mutable_data(pten::TransToFluidPlace(arg_def.backend),
                         pten::TransToProtoVarType(arg_def.dtype));
    return MakePtenDenseTensor(*tensor);
  } else if (variable->template IsType<framework::SelectedRows>()) {
    auto* tensor = variable->template GetMutable<framework::SelectedRows>();
    tensor->mutable_value()->mutable_data(
        pten::TransToFluidPlace(arg_def.backend),
        pten::TransToProtoVarType(arg_def.dtype));
    // TODO(chenweihang): adapt SelectedRows by xiaowei's design,
    // here the row and height will lost in output!
    return MakePtenDenseTensor(tensor->value());
  } else {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Unsupported shared output `%s` type now when call pt kernel.",
        framework::ToTypeName(variable->Type())));
  }
  return {};
}

void MovesStorage(pten::DenseTensor* src, paddle::framework::Tensor* dst) {
  PADDLE_ENFORCE_NOT_NULL(
      src,
      platform::errors::InvalidArgument(
          "The source DenseTensor is nullptr when move storage."));
  PADDLE_ENFORCE_NOT_NULL(
      dst,
      platform::errors::InvalidArgument(
          "The destination Tensor is nullptr when move storage."));
  dst->Resize(src->dims());
  auto storage = src->release();
  std::shared_ptr<paddle::memory::allocation::Allocation> holder(
      new TensorStorage(std::move(storage)));
  dst->ResetHolderWithType(holder, pten::TransToProtoVarType(src->dtype()));
}

void MovesStorage(pten::DenseTensor* src, paddle::framework::LoDTensor* dst) {
  PADDLE_ENFORCE_NOT_NULL(
      src,
      platform::errors::InvalidArgument(
          "The source DenseTensor is nullptr when move storage."));
  PADDLE_ENFORCE_NOT_NULL(
      dst,
      platform::errors::InvalidArgument(
          "The destination LoDTensor is nullptr when move storage."));
  SetLoD(dst->mutable_lod(), src->lod());
  MovesStorage(src, static_cast<paddle::framework::Tensor*>(dst));
}

void ReMakePtenDenseTensor(const paddle::framework::Tensor& src,
                           pten::DenseTensor* dst) {
  auto* meta = pten::CompatibleDenseTensorUtils::GetMutableMeta(dst);
  meta->dims = src.dims();
  // Since the type of DenseTensorMeta is const, const_cast must be used
  const_cast<DataType&>(meta->dtype) = pten::TransToPtenDataType(src.type());
  // Since the type of DenseTensorMeta is const, const_cast must be used
  const_cast<DataLayout&>(meta->layout) =
      pten::TransToPtenDataLayout(src.layout());
  auto* shared_storage = static_cast<SharedStorage*>(
      pten::CompatibleDenseTensorUtils::UnsafeGetMutableStorage(dst));
  PADDLE_ENFORCE_NOT_NULL(
      shared_storage,
      platform::errors::NotFound(
          "Target DenseTensor's shared storage is nullptr."));
  shared_storage->ResetAllocation(src.Holder(), src.offset());
}

void ReMakePtenDenseTensor(const paddle::framework::LoDTensor& src,
                           pten::DenseTensor* dst) {
  auto* meta = pten::CompatibleDenseTensorUtils::GetMutableMeta(dst);
  meta->dims = src.dims();
  // Since the type of DenseTensorMeta is const, const_cast must be used
  const_cast<DataType&>(meta->dtype) = pten::TransToPtenDataType(src.type());
  // Since the type of DenseTensorMeta is const, const_cast must be used
  const_cast<DataLayout&>(meta->layout) =
      pten::TransToPtenDataLayout(src.layout());
  SetLoD(&(meta->lod), src.lod());
  auto* shared_storage = static_cast<SharedStorage*>(
      pten::CompatibleDenseTensorUtils::UnsafeGetMutableStorage(dst));
  PADDLE_ENFORCE_NOT_NULL(
      shared_storage,
      platform::errors::NotFound(
          "Target DenseTensor's shared storage is nullptr."));
  shared_storage->ResetAllocation(src.Holder(), src.offset());
}

void ReMakePtenDenseTensorFromVar(const framework::Variable& variable,
                                  const pten::TensorArgDef& arg_def,
                                  pten::DenseTensor* dst) {
  auto expected_place = pten::TransToFluidPlace(arg_def.backend);

  if (variable.IsType<framework::LoDTensor>()) {
    const auto& tensor = variable.Get<framework::LoDTensor>();
    if (!platform::is_same_place(tensor.place(), expected_place)) {
      framework::LoDTensor tmp_tensor;
      framework::TensorCopySync(tensor, expected_place, &tmp_tensor);
      ReMakePtenDenseTensor(tmp_tensor, dst);
    } else {
      ReMakePtenDenseTensor(tensor, dst);
    }
  } else if (variable.IsType<framework::SelectedRows>()) {
    // TODO(chenweihang): now we don't deal with row and height
    // by xiaowei's advice
    const auto& tensor = variable.Get<framework::SelectedRows>();
    if (!platform::is_same_place(tensor.value().place(), expected_place)) {
      framework::Tensor tmp_tensor;
      TensorCopySync(tensor.value(), expected_place, &tmp_tensor);
      // TODO(chenweihang): adapt SelectedRows by xiaowei's design
      ReMakePtenDenseTensor(tmp_tensor, dst);
    } else {
      ReMakePtenDenseTensor(tensor.value(), dst);
    }
  } else {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Unsupported shared input `%s` type now when call pt kernel.",
        framework::ToTypeName(variable.Type())));
  }
}

void ReMakePtenDenseTensorFromVar(framework::Variable* variable,
                                  const pten::TensorArgDef& arg_def,
                                  pten::DenseTensor* dst) {
  // mutable_data before run kernel, to avoid share output form
  // KernelContext to original tensor
  if (variable->template IsType<framework::LoDTensor>()) {
    auto* tensor = variable->template GetMutable<framework::LoDTensor>();
    // TODO(chenweihang): use original var type if arg_def.dtype is UNDEFINED
    tensor->mutable_data(pten::TransToFluidPlace(arg_def.backend),
                         pten::TransToProtoVarType(arg_def.dtype));
    ReMakePtenDenseTensor(*tensor, dst);
  } else if (variable->template IsType<framework::SelectedRows>()) {
    auto* tensor = variable->template GetMutable<framework::SelectedRows>();
    tensor->mutable_value()->mutable_data(
        pten::TransToFluidPlace(arg_def.backend),
        pten::TransToProtoVarType(arg_def.dtype));
    // TODO(chenweihang): adapt SelectedRows by xiaowei's design,
    // here the row and height will lost in output!
    ReMakePtenDenseTensor(tensor->value(), dst);
  } else {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Unsupported shared output `%s` type now when call pt kernel.",
        framework::ToTypeName(variable->Type())));
  }
}

}  // namespace experimental
}  // namespace paddle
