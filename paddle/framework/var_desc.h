/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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

#include <vector>
#include "glog/logging.h"
#include "paddle/framework/framework.pb.h"

namespace paddle {
namespace framework {

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

class VarDescBind {
 public:
  explicit VarDescBind(const std::string &name) {
    desc_.set_name(name);
    desc_.set_type(VarDesc::LOD_TENSOR);
  }

  explicit VarDescBind(const VarDesc &desc) : desc_(desc) {}

  VarDesc *Proto() { return &desc_; }

  std::string Name() const { return desc_.name(); }

  void SetShape(const std::vector<int64_t> &dims);

  void SetDataType(DataType data_type);

  std::vector<int64_t> Shape() const;

  DataType GetDataType() const;

  void SetLoDLevel(int32_t lod_level);

  int32_t GetLodLevel() const;

  VarDesc::VarType GetType() const;

  void SetType(VarDesc::VarType type);

  bool Persistable() const { return desc_.persistable(); }

  void SetPersistable(bool persistable) { desc_.set_persistable(persistable); }

 private:
  const TensorDesc &tensor_desc() const;
  TensorDesc *mutable_tensor_desc();

  VarDesc desc_;
};
}  // namespace framework
}  // namespace paddle
