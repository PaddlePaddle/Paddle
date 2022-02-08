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

#include "paddle/pten/core/tensor_utils.h"

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

pten::Scalar MakePtenScalar(const paddle::framework::Tensor& src) {
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
  auto expected_place = pten::TransToPtenPlace(pten::Backend::CPU);
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

pten::ScalarArray MakePtenScalarArray(const paddle::framework::Tensor& src) {
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
  auto expected_place = pten::TransToPtenPlace(pten::Backend::CPU);
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
  auto expected_place = pten::TransToPtenPlace(pten::Backend::CPU);

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

void ResetTensorDtypeAndLayoutByArgDef(pten::TensorBase* dst,
                                       const pten::TensorArgDef& arg_def) {
  VLOG(5) << "ResetTensor by TensorArgDef.";
  if (pten::DenseTensor::classof(dst)) {
    auto* dense_t = static_cast<pten::DenseTensor*>(dst);
    if (dense_t->initialized()) {
      return;
    }
    auto* meta = pten::DenseTensorUtils::GetMutableMeta(dense_t);
    meta->dtype = arg_def.dtype;
    meta->layout = arg_def.layout;
  } else if (pten::SelectedRows::classof(dst)) {
    auto* selected_rows = static_cast<pten::SelectedRows*>(dst);
    auto* meta =
        pten::DenseTensorUtils::GetMutableMeta(selected_rows->mutable_value());
    meta->dtype = arg_def.dtype;
    meta->layout = arg_def.layout;
  } else {
    PADDLE_THROW(pten::errors::Unimplemented(
        "Unsupported tensor type is received when reseting tensor dtype and "
        "layout by argument definition."));
  }
}

}  // namespace experimental
}  // namespace paddle
