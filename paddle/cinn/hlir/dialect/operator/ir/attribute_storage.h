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
#include "paddle/cinn/operator_fusion/fusion_tracker/tracker.h"
#include "paddle/pir/include/core/attribute_base.h"
#include "paddle/pir/include/core/operation.h"
#include "paddle/pir/include/dialect/shape/utils/dim_expr.h"

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
  std::unordered_map<::pir::Operation*,
                     std::vector<cinn::hlir::framework::pir::ScheduleInfoNode>>
      alignment_schedule_info;
  std::vector<int64_t> reduce_axis;
  std::vector<int64_t> loop_ranges;
  std::vector<symbol::DimExpr> loop_ranges_expr;

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
    size_t hash_value = std::hash<std::string>{}(key.group_id);

    for (auto op : key.ops) {
      hash_value =
          pir::detail::hash_combine(hash_value, std::hash<void*>()(op));
    }

    for (auto d : key.loop_ranges) {
      hash_value =
          pir::detail::hash_combine(hash_value, std::hash<int64_t>()(d));
    }

    for (auto d : key.reduce_axis) {
      hash_value =
          pir::detail::hash_combine(hash_value, std::hash<int64_t>()(d));
    }
    return hash_value;
  }

  bool operator==(const ParamKey& key) const {
    return data_.group_id == key.group_id;
  }

  const ParamKey& GetAsKey() const { return data_; }

 private:
  ParamKey data_;
};

struct FusionTrackerPtrAttributeStorage : public pir::AttributeStorage {
  using ParamKey = cinn::fusion::FusionTrackerPtr;
  explicit FusionTrackerPtrAttributeStorage(const ParamKey& key) : data_(key) {}

  static FusionTrackerPtrAttributeStorage* Construct(const ParamKey& key) {
    return new FusionTrackerPtrAttributeStorage(key);
  }

  static std::size_t HashValue(const ParamKey& key) {
    return std::hash<ParamKey>()(key);
  }

  bool operator==(const ParamKey& key) const { return data_ == key; }

  const ParamKey& GetAsKey() const { return data_; }

 private:
  ParamKey data_;
};

struct CINNKernelInfoAttributeStorage : public pir::AttributeStorage {
  using ParamKey = cinn::hlir::framework::pir::CINNKernelInfo;
  explicit CINNKernelInfoAttributeStorage(const ParamKey& key) : data_(key) {}

  static CINNKernelInfoAttributeStorage* Construct(const ParamKey& key) {
    return new CINNKernelInfoAttributeStorage(key);
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
