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
#include <string>
#include <vector>
#include "paddle/fluid/lite/model_parser/desc_apis.h"

namespace paddle {
namespace lite {
namespace cpp {

/*
 * The cpp::VarDesc is the internal representation for Var. All the internal
 * imprementation should use it, not the pb::VarDesc.
 */
class VarDesc : public VarDescAPI {
 protected:
  std::string name_;
  bool persistable_;
  std::vector<int64_t> shape_;
  VarDataType type_;
  VarDataType data_type_;

 public:
  VarDesc() = default;
  explicit VarDesc(const std::string& name) : name_(name) {}

  std::string Name() const override { return name_; }

  void SetName(const std::string& name) override { name_ = name; }

  bool Persistable() const override { return persistable_; }

  void SetPersistable(bool persistable) override { persistable_ = persistable; }

  std::vector<int64_t> Shape() const override { return shape_; }

  void SetShape(const std::vector<int64_t>& shape) override {
    shape_.assign(shape.begin(), shape.end());
  }

  VarDataType Type() const override { return type_; }

  void SetType(VarDataType type) override { type_ = type; }

  VarDataType DataType() const override { return data_type_; }

  void SetDataType(VarDataType type) override { data_type_ = type; }
};

}  // namespace cpp
}  // namespace lite
}  // namespace paddle
