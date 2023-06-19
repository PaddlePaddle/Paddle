// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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
#include "paddle/cinn/frontend/paddle/cpp/desc_api.h"

namespace cinn::frontend::paddle::cpp {

/*
 * The cpp::VarDesc is the internal representation for Op. All the internal
 * imprementation should use it, not the pb::VarDesc.
 */
class VarDesc : public VarDescAPI {
 public:
  VarDesc() = default;

  explicit VarDesc(std::string name) : name_(name) {}

  std::string Name() const override { return name_; }

  void SetName(std::string name) override { name_ = name; }

  Type GetType() const override { return type_; }

  void SetType(Type type) override { type_ = type; }

  bool Persistable() const override { return persistable_; }

  void SetPersistable(bool persistable) override { persistable_ = persistable; }

  Type GetDataType() const { return data_type_; }

  void SetDataType(Type data_type) { data_type_ = data_type; }

  void SetShape(const std::vector<int64_t> &dims) { shape_ = dims; }

  std::vector<int64_t> GetShape() const { return shape_; }

 private:
  std::string name_;
  Type type_;
  Type data_type_;
  bool persistable_;
  std::vector<int64_t> shape_;
};

}  // namespace cinn::frontend::paddle::cpp
