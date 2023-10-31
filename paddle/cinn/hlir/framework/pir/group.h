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
#include <string>
#include <vector>

#include "paddle/cinn/hlir/framework/op.h"
#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/pir/core/operation.h"

namespace cinn {
namespace hlir {
namespace framework {
namespace pir {
using framework::OpPatternKind;

// TODO(Aurelius84): Need to be replaced with CinnGroupOp
struct Group {
 public:
  Group() = default;
  explicit Group(const std::vector<::pir::Operation*>& group_ops)
      : ops(group_ops) {}

  explicit Group(std::initializer_list<::pir::Operation*> group_ops)
      : ops(group_ops) {}

  // distance to last group.
  int depth{0};
  int max_depth{0};
  int min_depth{INT_MAX};
  // group id, consisted of op's id.
  std::string group_id{""};
  // global unique id.
  std::string unique_id{"uniq"};
  // op in this group
  std::vector<::pir::Operation*> ops;
  std::unordered_set<::pir::Operation*> ops_set;
  // input ops of the group.
  std::unordered_map<::pir::Operation*, int> input_ops;
  // output ops of the group.
  std::unordered_set<::pir::Operation*> output_ops;
  // op pattern kind.
  OpPatternKind op_pattern_kind{kReduction};
  // internal op, the output is used by multi-op.
  // internal op can't use compute inline, should use buffer.
  std::unordered_set<::pir::Operation*> internal_ops;
  // master op for schedule
  std::unordered_set<::pir::Operation*> master_ops;

  // fused sub-groups, used for fusion merge pass
  std::vector<std::shared_ptr<Group>> fused_sub_groups;
  // if as sub-group, used for belong groups.
  std::unordered_set<std::shared_ptr<Group>> belong_groups;

  // for op lowering.
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;

  struct SharedGroupHasher {
    size_t operator()(const std::shared_ptr<Group>& group) const noexcept {
      return std::hash<uint64_t>()(reinterpret_cast<uint64_t>(group.get()));
    }
  };
  struct SharedGroupComparator {
    bool operator()(const std::shared_ptr<Group>& first,
                    const std::shared_ptr<Group>& second) const noexcept {
      return first.get() == second.get();
    }
  };

  std::vector<::pir::Operation*> CollectOps() {
    if (fused_sub_groups.size()) {
      std::vector<::pir::Operation*> tmp_ops;
      for (auto& group : fused_sub_groups) {
        tmp_ops.insert(tmp_ops.end(), group->ops.begin(), group->ops.end());
      }
      return tmp_ops;
    } else {
      return ops;
    }
  }

  void WalkOps(const std::function<void(::pir::Operation*)>& VisitOp) const {
    if (fused_sub_groups.size()) {
      for (auto& group : fused_sub_groups) {
        for (const auto& op : group->ops) {
          VisitOp(op);
        }
      }
    } else {
      for (const auto& op : ops) {
        VisitOp(op);
      }
    }
  }

  std::unordered_set<::pir::Operation*> OpSet() {
    std::unordered_set<::pir::Operation*> op_set;
    for (auto op : CollectOps()) {
      op_set.insert(op);
    }
    return op_set;
  }

  // TODO(phlrain) : impliment GetInputNodeDatas GetOutputNodeDatas func
  // std::unordered_set<::pir::Value> GetInputNodeDatas() { return {}; }
  // std::unordered_set<::pir::Value> GetOutputNodeDatas() { return {}; }

  std::string GetFuncName() { return "fn_" + group_id + unique_id; }

 public:
  const std::unordered_set<std::shared_ptr<Group>,
                           SharedGroupHasher,
                           SharedGroupComparator>&
  producer_groups() const {
    return producer_groups_;
  }

  const std::unordered_set<std::shared_ptr<Group>,
                           SharedGroupHasher,
                           SharedGroupComparator>&
  consumer_groups() const {
    return consumer_groups_;
  }

  std::unordered_set<std::shared_ptr<Group>,
                     SharedGroupHasher,
                     SharedGroupComparator>*
  mut_producer_groups() {
    return &producer_groups_;
  }

  std::unordered_set<std::shared_ptr<Group>,
                     SharedGroupHasher,
                     SharedGroupComparator>*
  mut_consumer_groups() {
    return &consumer_groups_;
  }

  OpPatternKind kind() const { return op_pattern_kind; }

  std::string FuncName() const { return CompatibleInfo::GroupOpsName(ops); }

 private:
  // input groups
  std::unordered_set<std::shared_ptr<Group>,
                     SharedGroupHasher,
                     SharedGroupComparator>
      producer_groups_;
  // output grous
  std::unordered_set<std::shared_ptr<Group>,
                     SharedGroupHasher,
                     SharedGroupComparator>
      consumer_groups_;
};

}  // namespace pir
}  // namespace framework
}  // namespace hlir
}  // namespace cinn
