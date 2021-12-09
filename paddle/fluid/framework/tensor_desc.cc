// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/framework/tensor_desc.h"

namespace paddle {
namespace framework {

// template<typename T>
// const std::vector<T>& ExtractTensorDescValue(const TensorDescValue& desc_val)
// {
//   // PADDLE_ENFORCE_EQ(desc_val.type(), typeid(std::vector<T>),
//   //                   platform::errors::PreconditionNotMet(
//   //                       "Found mismatch type in
//   ExtractTensorDescValue."));
//   return BOOST_GET_CONST(std::vector<T>, desc_val);
// }

template <typename T, typename RepeatedField>
inline static void VectorToRepeated(const std::vector<T> &vec,
                                    RepeatedField *repeated_field) {
  repeated_field->Clear();
  repeated_field->Reserve(vec.size());
  for (const auto &elem : vec) {
    *repeated_field->Add() = elem;
  }
}

// convert between std::vector and protobuf repeated.
template <typename T>
inline static std::vector<T> RepeatedToVector(
    const google::protobuf::RepeatedField<T> &repeated_field) {
  std::vector<T> ret;
  ret.reserve(repeated_field.size());
  std::copy(repeated_field.begin(), repeated_field.end(),
            std::back_inserter(ret));
  return ret;
}

TensorDescValue GetTensorDescValue(
    const proto::VarType::TensorDesc &tensor_desc) {
  switch (tensor_desc.data_type()) {
    case proto::VarType::Type::VarType_Type_INT32: {
      return RepeatedToVector(tensor_desc.int32_val());
    }
    case proto::VarType::Type::VarType_Type_INT64: {
      return RepeatedToVector(tensor_desc.int64_val());
    }
    case proto::VarType::Type::VarType_Type_FP32: {
      return RepeatedToVector(tensor_desc.float_val());
    }
    // case proto::VarType::Type::VarType_Type_STRING: {
    //   return RepeatedToVector(tensor_desc.string_val());
    // }

    default:
      PADDLE_THROW(platform::errors::Unavailable(
          "Unsupport TensorDesc type %d.", tensor_desc.data_type()));
  }

  return boost::blank();
}

void SetTensorDescValue(proto::VarType::TensorDesc *tensor_desc,
                        const TensorDescValue &val) {
  switch (tensor_desc->data_type()) {
    case proto::VarType::Type::VarType_Type_INT32: {
      auto &vec = ExtractTensorDescValue<std::vector<int>>(val);
      VectorToRepeated(vec, tensor_desc->mutable_int32_val());
    }
    case proto::VarType::Type::VarType_Type_INT64: {
      auto &vec = ExtractTensorDescValue<std::vector<int64_t>>(val);
      VectorToRepeated(vec, tensor_desc->mutable_int64_val());
    }
    case proto::VarType::Type::VarType_Type_FP32: {
      auto &vec = ExtractTensorDescValue<std::vector<float>>(val);
      VectorToRepeated(vec, tensor_desc->mutable_float_val());
    }
    // case proto::VarType::Type::VarType_Type_STRING: {
    //   auto &vec = ExtractTensorDescValue<std::vector<std::string>>(val);
    //   VectorToRepeated(vec, tensor_desc->mutable_string_val());
    // }
    default:
      PADDLE_THROW(platform::errors::Unavailable(
          "Unsupport TensorDesc type %d.", tensor_desc->data_type()));
  }
}

}  // namespace framework
}  // namespace paddle
