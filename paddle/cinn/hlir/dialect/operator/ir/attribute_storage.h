// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "paddle/cinn/hlir/framework/op.h"
#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/pir/core/attribute_base.h"
#include "paddle/pir/core/operation.h"

namespace cinn {
namespace dialect {

// TODO(Aurelius84): Need to figure out what we need indeed for GroupOp.
// Currently we paste almost members here and will remove them step by
// step.
struct GroupInfo {
 public:
  explicit GroupInfo(const std::vector<::pir::Operation*>& group_ops)
      : ops(group_ops) {
    Initialize();
  }

  explicit GroupInfo(std::initializer_list<::pir::Operation*> group_ops)
      : ops(group_ops) {
    Initialize();
  }

  std::string group_id;
  std::string fn_name;
  hlir::framework::OpPatternKind op_pattern_kind;
  std::vector<::pir::Operation*> ops;
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;

 private:
  void Initialize() {
    op_pattern_kind = hlir::framework::OpPatternKind::kElementWise;
    fn_name = hlir::framework::pir::CompatibleInfo::GroupOpsName(ops);
  }
};

struct GroupInfoAttributeStorage : public pir::AttributeStorage {
  using ParamKey = GroupInfo;
  explicit GroupInfoAttributeStorage(const ParamKey& key) : data_(key) {}

  static GroupInfoAttributeStorage* Construct(const ParamKey& key) {
    return new GroupInfoAttributeStorage(key);
  }

  static std::size_t HashValue(const ParamKey& key) {
    return std::hash<std::string>{}(key.group_id);
  }

  bool operator==(const ParamKey& key) const {
    return data_.group_id == key.group_id;
  }

  const ParamKey& GetAsKey() const { return data_; }

 private:
  ParamKey data_;
};

struct JITInfoAttributeStorage : public pir::AttributeStorage {
  using ParamKey = cinn::hlir::framework::pir::CUDAJITInfo;
  explicit JITInfoAttributeStorage(const ParamKey& key) : data_(key) {}

  static JITInfoAttributeStorage* Construct(const ParamKey& key) {
    return new JITInfoAttributeStorage(key);
  }

  static std::size_t HashValue(const ParamKey& key) {
    return std::hash<int64_t>()(*(reinterpret_cast<int64_t*>(key.fn_ptr)));
  }

  bool operator==(const ParamKey& key) const {
    return data_.fn_ptr == key.fn_ptr;
  }

  const ParamKey& GetAsKey() const { return data_; }

 private:
  ParamKey data_;
};

}  // namespace dialect
}  // namespace cinn
