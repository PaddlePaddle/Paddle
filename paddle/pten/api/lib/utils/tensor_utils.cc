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

#include <utility>
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

std::unique_ptr<pten::DenseTensor> MakePtenDenseTensor(
    const paddle::framework::Tensor& tensor,
    const pten::TensorArgDef& arg_def) {
  pten::DenseTensorMeta meta{arg_def.dtype,
                             tensor.dims(),
                             pten::TransToPtenDataLayout(tensor.layout())};

  if (tensor.IsInitialized() &&
      tensor.place() == pten::TransToFluidPlace(arg_def.backend)) {
    auto shared_storage =
        pten::make_intrusive<SharedStorage>(tensor.Holder(), tensor.offset());
    return std::make_unique<pten::DenseTensor>(std::move(shared_storage),
                                               std::move(meta));
  } else {
    return std::make_unique<pten::DenseTensor>(
        std::move(pten::make_intrusive<SharedStorage>(
            pten::TransToFluidPlace(arg_def.backend))),
        std::move(meta));
  }
}

std::unique_ptr<pten::DenseTensor> MakePtenDenseTensor(
    const paddle::framework::LoDTensor& tensor,
    const pten::TensorArgDef& arg_def) {
  pten::DenseTensorMeta meta{arg_def.dtype,
                             tensor.dims(),
                             pten::TransToPtenDataLayout(tensor.layout()),
                             pten::TransToPtenLoD(tensor.lod())};

  if (tensor.IsInitialized() &&
      tensor.place() == pten::TransToFluidPlace(arg_def.backend)) {
    auto shared_storage =
        pten::make_intrusive<SharedStorage>(tensor.Holder(), tensor.offset());
    return std::make_unique<pten::DenseTensor>(std::move(shared_storage),
                                               std::move(meta));
  } else {
    return std::make_unique<pten::DenseTensor>(
        std::move(pten::make_intrusive<SharedStorage>(
            pten::TransToFluidPlace(arg_def.backend))),
        std::move(meta));
  }
}

pten::Scalar MakePtenScalar(const paddle::framework::LoDTensor& src) {
  PADDLE_ENFORCE_EQ(src.numel(),
                    1,
                    paddle::platform::errors::InvalidArgument(
                        "The Scalar only supports Tensor with 1 element, "
                        "but now Tensor has %d element.",
                        src.numel()));
  switch (src.type()) {
    case paddle::framework::proto::VarType::FP32:
      return {src.template data<float>()[0]};
    case paddle::framework::proto::VarType::FP64:
      return {src.template data<double>()[0]};
    case paddle::framework::proto::VarType::FP16:
      return {src.template data<float16>()[0]};
    case paddle::framework::proto::VarType::BF16:
      return {src.template data<bfloat16>()[0]};
    case paddle::framework::proto::VarType::INT32:
      return {src.template data<int32_t>()[0]};
    case paddle::framework::proto::VarType::INT64:
      return {src.template data<int64_t>()[0]};
    case paddle::framework::proto::VarType::INT16:
      return {src.template data<int16_t>()[0]};
    case paddle::framework::proto::VarType::INT8:
      return {src.template data<int8_t>()[0]};
    case paddle::framework::proto::VarType::UINT8:
      return {src.template data<uint8_t>()[0]};
    case paddle::framework::proto::VarType::BOOL:
      return {src.template data<bool>()[0]};
    case paddle::framework::proto::VarType::COMPLEX64:
      return {src.template data<complex64>()[0]};
    case paddle::framework::proto::VarType::COMPLEX128:
      return {src.template data<complex128>()[0]};
    default:
      PADDLE_THROW(paddle::platform::errors::InvalidArgument(
          "Data type error. Don't support casting a %d LoDTensor to Scalar.",
          src.type()));
  }
}

pten::Scalar MakePtenScalarFromVar(const framework::Variable& variable) {
  auto expected_place = pten::TransToFluidPlace(pten::Backend::CPU);
  if (variable.IsType<framework::LoDTensor>()) {
    const auto& tensor = variable.Get<framework::LoDTensor>();
    if (!platform::is_same_place(tensor.place(), expected_place)) {
      framework::LoDTensor tmp_tensor;
      framework::TensorCopySync(tensor, expected_place, &tmp_tensor);
      return MakePtenScalar(tmp_tensor);
    } else {
      return MakePtenScalar(tensor);
    }
  } else {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Unsupport casting input `%s` type to Scalar when call pt "
        "kernel.",
        framework::ToTypeName(variable.Type())));
  }
}

pten::ScalarArray MakePtenScalarArray(const paddle::framework::LoDTensor& src) {
  if (src.type() == paddle::framework::proto::VarType::INT64) {
    return {src.data<int64_t>(), src.numel()};
  } else if (src.type() == paddle::framework::proto::VarType::INT32) {
    return {src.data<int32_t>(), src.numel()};
  } else {
    PADDLE_THROW(paddle::platform::errors::InvalidArgument(
        "Data type error. When cast a LoDTensor to ScalarArray, "
        "the data type of LoDTensor must be int32 or int64, "
        "but now data type is %s.",
        src.type()));
  }
}

pten::ScalarArray MakePtenScalarArrayFromVar(
    const framework::Variable& variable) {
  auto expected_place = pten::TransToFluidPlace(pten::Backend::CPU);
  if (variable.IsType<framework::LoDTensor>()) {
    const auto& tensor = variable.Get<framework::LoDTensor>();
    if (!platform::is_same_place(tensor.place(), expected_place)) {
      framework::LoDTensor tmp_tensor;
      framework::TensorCopySync(tensor, expected_place, &tmp_tensor);
      return MakePtenScalarArray(tmp_tensor);
    } else {
      return MakePtenScalarArray(tensor);
    }
  } else {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Unsupport casting input `%s` type to ScalarArray when call pt "
        "kernel.",
        framework::ToTypeName(variable.Type())));
  }
}

pten::ScalarArray MakePtenScalarArrayFromVarList(
    const std::vector<framework::Variable*>& variable_list) {
  if (variable_list.size() == 0) {
    return pten::ScalarArray();
  }
  auto expected_place = pten::TransToFluidPlace(pten::Backend::CPU);

  paddle::framework::proto::VarType::Type data_type;
  auto* first_var = variable_list.front();
  if (first_var->IsType<framework::LoDTensor>()) {
    const auto& tensor = first_var->Get<framework::LoDTensor>();
    data_type = tensor.type();
  } else {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Unsupport casting input `%s` type to VectorTensor when call pt "
        "kernel.",
        framework::ToTypeName(first_var->Type())));
  }

  std::vector<int64_t> vector_data;
  vector_data.reserve(variable_list.size());

  if (data_type == paddle::framework::proto::VarType::INT64) {
    for (auto* var : variable_list) {
      if (var->IsType<framework::LoDTensor>()) {
        const auto& tensor = var->Get<framework::LoDTensor>();
        if (!platform::is_same_place(tensor.place(), expected_place)) {
          framework::LoDTensor tmp_tensor;
          framework::TensorCopySync(tensor, expected_place, &tmp_tensor);
          vector_data.push_back(*tmp_tensor.data<int64_t>());
        } else {
          vector_data.push_back(*tensor.data<int64_t>());
        }
      } else {
        PADDLE_THROW(platform::errors::Unimplemented(
            "Unsupport casting input `%s` type to VectorTensor when call pt "
            "kernel.",
            framework::ToTypeName(var->Type())));
      }
    }

  } else if (data_type == paddle::framework::proto::VarType::INT32) {
    for (auto* var : variable_list) {
      if (var->IsType<framework::LoDTensor>()) {
        const auto& tensor = var->Get<framework::LoDTensor>();
        if (!platform::is_same_place(tensor.place(), expected_place)) {
          framework::LoDTensor tmp_tensor;
          framework::TensorCopySync(tensor, expected_place, &tmp_tensor);
          vector_data.push_back(*tmp_tensor.data<int32_t>());
        } else {
          vector_data.push_back(*tensor.data<int32_t>());
        }
      } else {
        PADDLE_THROW(platform::errors::Unimplemented(
            "Unsupport casting input `%s` type to VectorTensor when call pt "
            "kernel.",
            framework::ToTypeName(var->Type())));
      }
    }
  } else {
    PADDLE_THROW(paddle::platform::errors::InvalidArgument(
        "Data type error. When cast a LoDTensor to VectorTensor, "
        "the data type of LoDTensor must be int32 or int64, "
        "but now data type is %s.",
        data_type));
  }

  return {vector_data};
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
    return MakePtenDenseTensor(*tensor, arg_def);
  } else if (variable->template IsType<framework::SelectedRows>()) {
    auto* tensor = variable->template GetMutable<framework::SelectedRows>();
    // TODO(chenweihang): adapt SelectedRows by xiaowei's design,
    // here the row and height will lost in output!
    return MakePtenDenseTensor(tensor->value(), arg_def);
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
  dst->set_type(pten::TransToProtoVarType(src->dtype()));
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

void MovesSharedStorage(pten::DenseTensor* src,
                        paddle::framework::Tensor* dst) {
  PADDLE_ENFORCE_NOT_NULL(
      src,
      platform::errors::InvalidArgument(
          "The source DenseTensor is nullptr when move allocation."));
  PADDLE_ENFORCE_NOT_NULL(
      dst,
      platform::errors::InvalidArgument(
          "The destination Tensor is nullptr when move allocation."));
  dst->Resize(src->dims());
  auto* storage = static_cast<SharedStorage*>(
      pten::CompatibleDenseTensorUtils::UnsafeGetMutableStorage(src));
  dst->ResetHolderWithType(storage->GetAllocation(),
                           pten::TransToProtoVarType(src->dtype()));
}

void MovesSharedStorage(pten::DenseTensor* src,
                        paddle::framework::LoDTensor* dst) {
  MovesSharedStorage(src, static_cast<paddle::framework::Tensor*>(dst));
  SetLoD(dst->mutable_lod(), src->lod());
}

void ReMakePtenDenseTensor(const paddle::framework::Tensor& src,
                           const pten::TensorArgDef& arg_def,
                           pten::DenseTensor* dst) {
  auto* meta = pten::CompatibleDenseTensorUtils::GetMutableMeta(dst);
  meta->dims = src.dims();
  // Since the type of DenseTensorMeta is const, const_cast must be used
  const_cast<DataType&>(meta->dtype) = arg_def.dtype;
  // Since the type of DenseTensorMeta is const, const_cast must be used
  const_cast<DataLayout&>(meta->layout) =
      pten::TransToPtenDataLayout(src.layout());

  auto* shared_storage = static_cast<SharedStorage*>(
      pten::CompatibleDenseTensorUtils::UnsafeGetMutableStorage(dst));
  PADDLE_ENFORCE_NOT_NULL(
      shared_storage,
      platform::errors::NotFound(
          "Target DenseTensor's shared storage is nullptr."));

  if (src.IsInitialized()) {
    shared_storage->ResetAllocation(src.Holder(), src.offset());
  }
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

  auto* shared_storage = static_cast<SharedStorage*>(
      pten::CompatibleDenseTensorUtils::UnsafeGetMutableStorage(dst));
  PADDLE_ENFORCE_NOT_NULL(
      shared_storage,
      platform::errors::NotFound(
          "Target DenseTensor's shared storage is nullptr."));

  if (src.IsInitialized()) {
    shared_storage->ResetAllocation(src.Holder(), src.offset());
  }
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

  if (src.IsInitialized()) {
    shared_storage->ResetAllocation(src.Holder(), src.offset());
  }
}

void ReMakePtenDenseTensor(const paddle::framework::LoDTensor& src,
                           const pten::TensorArgDef& arg_def,
                           pten::DenseTensor* dst) {
  auto* meta = pten::CompatibleDenseTensorUtils::GetMutableMeta(dst);
  meta->dims = src.dims();
  // Since the type of DenseTensorMeta is const, const_cast must be used
  const_cast<DataType&>(meta->dtype) = arg_def.dtype;
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
  if (src.IsInitialized() &&
      src.place() == pten::TransToFluidPlace(arg_def.backend)) {
    shared_storage->ResetAllocation(src.Holder(), src.offset());
  } else {
    shared_storage->ResetAllocationPlace(
        pten::TransToFluidPlace(arg_def.backend));
  }
}

void ReMakePtenDenseTensorFromVar(const framework::Variable& variable,
                                  const pten::TensorArgDef& arg_def,
                                  pten::DenseTensor* dst) {
  auto expected_place = pten::TransToFluidPlace(arg_def.backend);
  if (variable.IsType<framework::LoDTensor>()) {
    const auto& tensor = variable.Get<framework::LoDTensor>();
    // check input dtype before ReMakePtenDenseTensor
    PADDLE_ENFORCE(
        (arg_def.dtype == pten::TransToPtenDataType(tensor.type())),
        paddle::platform::errors::InvalidArgument(
            "The type of input data is diffrent from the type of the "
            "argument's definition in kernel."));
    if (!platform::is_same_place(tensor.place(), expected_place)) {
      framework::LoDTensor tmp_tensor;
      framework::TensorCopySync(tensor, expected_place, &tmp_tensor);
      ReMakePtenDenseTensor(tmp_tensor, arg_def, dst);
    } else {
      ReMakePtenDenseTensor(tensor, arg_def, dst);
    }
  } else if (variable.IsType<framework::SelectedRows>()) {
    // TODO(chenweihang): now we don't deal with row and height
    // by xiaowei's advice
    const auto& tensor = variable.Get<framework::SelectedRows>();
    PADDLE_ENFORCE(
        (arg_def.dtype == pten::TransToPtenDataType(tensor.value().type())),
        paddle::platform::errors::InvalidArgument(
            "The type of input data is diffrent from the type of the "
            "argument's definition in kernel."));
    if (!platform::is_same_place(tensor.value().place(), expected_place)) {
      framework::Tensor tmp_tensor;
      TensorCopySync(tensor.value(), expected_place, &tmp_tensor);
      // TODO(chenweihang): adapt SelectedRows by xiaowei's design
      ReMakePtenDenseTensor(tmp_tensor, arg_def, dst);
    } else {
      ReMakePtenDenseTensor(tensor.value(), arg_def, dst);
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
    ReMakePtenDenseTensor(*tensor, arg_def, dst);
  } else if (variable->template IsType<framework::SelectedRows>()) {
    auto* tensor = variable->template GetMutable<framework::SelectedRows>();
    // TODO(chenweihang): adapt SelectedRows by xiaowei's design,
    // here the row and height will lost in output!
    ReMakePtenDenseTensor(tensor->value(), arg_def, dst);
  } else {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Unsupported shared output `%s` type now when call pt kernel.",
        framework::ToTypeName(variable->Type())));
  }
}

static bool IsSameAllocation(const std::shared_ptr<memory::Allocation>& a,
                             const std::shared_ptr<memory::Allocation>& b) {
  return a->ptr() == b->ptr() && a->size() == b->size() &&
         platform::is_same_place(a->place(), b->place());
}

void MakeVariableFromPtenTensor(pten::DenseTensor* src,
                                framework::Variable* variable) {
  if (variable->IsType<framework::LoDTensor>()) {
    auto* tensor = variable->GetMutable<framework::LoDTensor>();

    auto dtype = pten::TransToProtoVarType(src->dtype());
    tensor->Resize(src->dims());
    SetLoD(tensor->mutable_lod(), src->lod());

    // here dynamic_cast is slow
    auto* storage = static_cast<SharedStorage*>(
        pten::CompatibleDenseTensorUtils::UnsafeGetMutableStorage(src));

    if (!tensor->IsInitialized() ||
        (tensor->IsInitialized() &&
         !IsSameAllocation(tensor->Holder(), storage->GetAllocation()))) {
      tensor->ResetHolderWithType(std::move(storage->GetAllocation()), dtype);
    } else {
      // Even the pten tensor and Variable have the same Alloctation (both have
      // the same pointer address, same size and same place)
      // but there is possible that they do not have the same data_type.
      // so, here we set the variable's type with the pten tensor dtype.
      tensor->set_type(dtype);
    }

  } else if (variable->IsType<framework::SelectedRows>()) {
    auto* tensor = variable->GetMutable<framework::SelectedRows>();
    auto dtype = pten::TransToProtoVarType(src->dtype());

    if (!tensor->value().IsInitialized()) {
      auto storage = dynamic_cast<SharedStorage*>(
          pten::CompatibleDenseTensorUtils::UnsafeGetMutableStorage(src));
      tensor->mutable_value()->ResetHolderWithType(
          std::move(storage->GetAllocation()), dtype);
    }
  } else {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Unsupported shared input `%s` type now when call pt kernel.",
        framework::ToTypeName(variable->Type())));
  }
}

}  // namespace experimental
}  // namespace paddle
