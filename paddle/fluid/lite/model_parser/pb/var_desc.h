// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <glog/logging.h>
#include <algorithm>
#include <string>
#include <vector>
#include "paddle/fluid/framework/framework.pb.h"

namespace paddle {
namespace lite {
namespace pb {

// convert between std::vector and protobuf repeated.
template <typename T>
inline std::vector<T> RepeatedToVector(
    const google::protobuf::RepeatedField<T> &repeated_field) {
  std::vector<T> ret;
  ret.reserve(repeated_field.size());
  std::copy(repeated_field.begin(), repeated_field.end(),
            std::back_inserter(ret));
  return ret;
}

template <typename T, typename RepeatedField>
inline void VectorToRepeated(const std::vector<T> &vec,
                             RepeatedField *repeated_field) {
  repeated_field->Clear();
  repeated_field->Reserve(vec.size());
  for (const auto &elem : vec) {
    *repeated_field->Add() = elem;
  }
}

// Specialize vector<bool>.
template <typename RepeatedField>
inline void VectorToRepeated(const std::vector<bool> &vec,
                             RepeatedField *repeated_field) {
  repeated_field->Clear();
  repeated_field->Reserve(vec.size());
  for (auto elem : vec) {
    *repeated_field->Add() = elem;
  }
}

class VarDesc {
 public:
  explicit VarDesc(const std::string &name) {
    desc_.set_name(name);
    // TODO(paddle-dev): Why default to lodtensor.
    desc_.mutable_type()->set_type(framework::proto::VarType::LOD_TENSOR);
  }

  explicit VarDesc(const framework::proto::VarDesc &desc) : desc_(desc) {}

  framework::proto::VarDesc *Proto() { return &desc_; }

  std::string Name() const { return desc_.name(); }

  void SetName(std::string name) { desc_.set_name(name); }

  void SetTensorDescNum(size_t num);

  size_t GetTensorDescNum() const;

  void SetShape(const std::vector<int64_t> &dims);

  void SetShapes(const std::vector<std::vector<int64_t>> &multiple_dims);

  std::vector<int64_t> GetShape() const;

  std::vector<std::vector<int64_t>> GetShapes() const;

  void SetDataType(framework::proto::VarType::Type data_type);

  void SetDataTypes(
      const std::vector<framework::proto::VarType::Type> &multiple_data_type);

  framework::proto::VarType::Type GetDataType() const;

  std::vector<framework::proto::VarType::Type> GetDataTypes() const;

  void SetLoDLevel(int32_t lod_level);

  void SetLoDLevels(const std::vector<int32_t> &multiple_lod_level);

  int32_t GetLoDLevel() const;

  std::vector<int32_t> GetLoDLevels() const;

  framework::proto::VarType::Type GetType() const;

  void SetType(framework::proto::VarType::Type type);

  bool Persistable() const { return desc_.persistable(); }

  void SetPersistable(bool persistable) { desc_.set_persistable(persistable); }

 private:
  const framework::proto::VarType::TensorDesc &tensor_desc() const;
  std::vector<framework::proto::VarType::TensorDesc> tensor_descs() const;
  framework::proto::VarType::TensorDesc *mutable_tensor_desc();
  std::vector<framework::proto::VarType::TensorDesc *> mutable_tensor_descs();

  framework::proto::VarDesc desc_;
};

}  // namespace pb
}  // namespace framework
}  // namespace paddle
