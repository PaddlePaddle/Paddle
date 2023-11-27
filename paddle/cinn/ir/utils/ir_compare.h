// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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
#include <vector>

#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_visitor.h"

namespace cinn {
namespace ir {
namespace ir_utils {

// Determine whether two ir AST trees are euqal by comparing their struct and
// fields of each node through dfs visitor
class IrEqualVisitor : public IRVisitorRequireReImpl<bool, const Expr*> {
 public:
  explicit IrEqualVisitor(bool allow_name_suffix_diff = false,
                          bool only_compare_structure = false)
      : allow_name_suffix_diff_(allow_name_suffix_diff),
        only_compare_structure_(only_compare_structure) {}
  // Return true if they are euqal, otherwise false;
  bool Compare(const Expr& lhs, const Expr& rhs);

 private:
  bool Compare(const std::string& lhs, const std::string& rhs);
  bool Compare(const std::map<std::string, attr_t>& lhs,
               const std::map<std::string, attr_t>& rhs);
  template <typename T>
  bool Compare(const std::vector<T>& lhs, const std::vector<T>& rhs);

#define __(op__) bool Visit(const op__* lhs, const Expr* other) override;
  NODETY_FORALL(__)
#undef __

  // whether allowing name suffix ends with "_[0-9]+" different
  bool allow_name_suffix_diff_ = false;
  // not compare name field of Expr
  bool only_compare_structure_ = false;
};

bool IRCompare(const Expr& lhs,
               const Expr& rhs,
               bool allow_name_suffix_diff = false);

}  // namespace ir_utils
}  // namespace ir
}  // namespace cinn
