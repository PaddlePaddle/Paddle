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
#include <set>
#include <string>
#include <vector>

#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_base.h"

namespace cinn {
namespace ir {
class Layout {
 public:
  std::string name_;
  std::string axis_names_;
  std::vector<ir::Var> axes_;

  Layout(const std::string& name, const std::vector<ir::Var>& axes)
      : name_(name), axes_(axes) {
    Verify();
  }

  explicit Layout(const std::string& name);

  inline const std::string& name() const { return name_; }
  // axis name without factor
  inline const std::string& axis_names() const { return axis_names_; }
  inline const std::vector<ir::Var>& axes() const { return axes_; }
  inline int ndims() const { return axes_.size(); }
  inline const Var operator[](int i) const { return axes_[i]; }
  inline const char axis_names(int i) const { return axis_names_[i]; }

  void Verify();
  Expr Make(const std::string& name, const std::vector<ir::Var>& axes);
};

}  // namespace ir
}  // namespace cinn
