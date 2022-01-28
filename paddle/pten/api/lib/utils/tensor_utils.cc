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
  return std::make_unique<pten::DenseTensor>(src);
}

pten::Scalar MakePtenScalarFromVar(const framework::Variable& variable) {
  auto expected_place = pten::TransToFluidPlace(pten::Backend::CPU);
  if (variable.IsType<framework::LoDTensor>()) {
    const auto& tensor = variable.Get<framework::LoDTensor>();
    if (!platform::is_same_place(tensor.place(), expected_place)) {
      framework::LoDTensor tmp_tensor;
      framework::TensorCopySync(tensor, expected_place, &tmp_tensor);
      return {tmp_tensor};
    } else {
      return {tensor};
    }
  } else {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Unsupport casting input `%s` type to Scalar when call pt "
        "kernel.",
        framework::ToTypeName(variable.Type())));
  }
}

pten::ScalarArray MakePtenScalarArray(const paddle::framework::Tensor& src) {
  return {src};
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

// TODO(chentianyu03): Inplace with ScalarArray constructor
pten::ScalarArray MakePtenScalarArrayFromVarList(
    const std::vector<framework::Variable*>& variable_list) {
  if (variable_list.size() == 0) {
    return pten::ScalarArray();
  }
  auto expected_place = pten::TransToFluidPlace(pten::Backend::CPU);

  std::vector<int64_t> vector_data;
  vector_data.reserve(variable_list.size());

  for (auto* var : variable_list) {
    paddle::framework::proto::VarType::Type data_type;
    if (var->IsType<framework::LoDTensor>()) {
      const auto& tensor = var->Get<framework::LoDTensor>();
      data_type = tensor.type();
      if (data_type == paddle::framework::proto::VarType::INT64) {
        const auto& tensor = var->Get<framework::LoDTensor>();
        if (tensor.IsInitialized() &&
            !platform::is_same_place(tensor.place(), expected_place)) {
          framework::LoDTensor tmp_tensor;
          framework::TensorCopySync(tensor, expected_place, &tmp_tensor);
          vector_data.push_back(*tmp_tensor.data<int64_t>());
        } else {
          vector_data.push_back(*tensor.data<int64_t>());
        }
      } else if (data_type == paddle::framework::proto::VarType::INT32) {
        const auto& tensor = var->Get<framework::LoDTensor>();
        if (tensor.IsInitialized() &&
            !platform::is_same_place(tensor.place(), expected_place)) {
          framework::LoDTensor tmp_tensor;
          framework::TensorCopySync(tensor, expected_place, &tmp_tensor);
          vector_data.push_back(*tmp_tensor.data<int32_t>());
        } else {
          vector_data.push_back(*tensor.data<int32_t>());
        }
      } else {
        PADDLE_THROW(paddle::platform::errors::InvalidArgument(
            "Data type error. When cast a LoDTensor to VectorTensor, "
            "the data type of LoDTensor must be int32 or int64, "
            "but now data type is %s.",
            data_type));
      }
    } else {
      PADDLE_THROW(platform::errors::Unimplemented(
          "Unsupport casting input `%s` type to VectorTensor when call pt "
          "kernel.",
          framework::ToTypeName(var->Type())));
    }
  }

  pten::ScalarArray result{vector_data};
  result.setInitByTensor(true);

  return result;
}

void SharesStorageBase(pten::DenseTensor* src, paddle::framework::Tensor* dst) {
  PADDLE_ENFORCE_NOT_NULL(
      src,
      platform::errors::InvalidArgument(
          "The source DenseTensor is nullptr when move allocation."));
  PADDLE_ENFORCE_NOT_NULL(
      dst,
      platform::errors::InvalidArgument(
          "The destination Tensor is nullptr when move allocation."));
  dst->Resize(src->dims());
  dst->ResetHolderWithType(src->Holder(),
                           pten::TransToProtoVarType(src->dtype()));
  dst->set_offset(src->meta().offset);
}

void SharesStorage(pten::DenseTensor* src, paddle::framework::Tensor* dst) {
  SharesStorageBase(src, static_cast<paddle::framework::Tensor*>(dst));
  SetLoD(dst->mutable_lod(), src->lod());
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

    if (!tensor->IsInitialized() ||
        (tensor->IsInitialized() &&
         !IsSameAllocation(tensor->Holder(), src->Holder()))) {
      tensor->ResetHolderWithType(std::move(src->Holder()), dtype);
    } else {
      // Even the pten tensor and Variable have the same Alloctation (both have
      // the same pointer address, same size and same place)
      // but there is possible that they do not have the same data_type.
      // so, here we set the variable's type with the pten tensor dtype.
      tensor->set_type(dtype);
    }

  } else if (variable->IsType<pten::SelectedRows>()) {
    auto* tensor = variable->GetMutable<pten::SelectedRows>();
    auto dtype = pten::TransToProtoVarType(src->dtype());

    if (!tensor->value().IsInitialized()) {
      tensor->mutable_value()->ResetHolderWithType(std::move(src->Holder()),
                                                   dtype);
    }
  } else {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Unsupported shared input `%s` type now when call pt kernel.",
        framework::ToTypeName(variable->Type())));
  }
}

void ResetTensorByArgDef(pten::DenseTensor* dst,
                         const pten::TensorArgDef& arg_def) {
  VLOG(5) << "ResetTensor by TensorArgDef.";
  auto* meta = pten::CompatibleDenseTensorUtils::GetMutableMeta(dst);
  meta->dtype = arg_def.dtype;
  meta->layout = arg_def.layout;
}

}  // namespace experimental
}  // namespace paddle
