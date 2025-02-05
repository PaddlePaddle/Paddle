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

#include "paddle/cinn/ir/layout.h"
#include "paddle/common/enforce.h"
namespace cinn {
namespace ir {

void Layout::Verify() {
  {
    PADDLE_ENFORCE_NE(
        name_.empty(),
        true,
        ::common::errors::InvalidArgument("The name should not be empty."));
    PADDLE_ENFORCE_NE(
        axes_.empty(),
        true,
        ::common::errors::InvalidArgument("The axes should not be empty."));
    axis_names_ = "";
    for (auto& axis : axes_) {
      PADDLE_ENFORCE_EQ(
          axis->name.size(),
          1U,
          ::common::errors::InvalidArgument("axis name size must be 1"));
      auto axis_name = axis->name[0];
      PADDLE_ENFORCE_EQ(
          (axis_name >= 'A' && axis_name <= 'Z') ||
              (axis_name >= 'a' && axis_name <= 'z'),
          true,
          ::common::errors::InvalidArgument("Axis name must be a letter."));
      PADDLE_ENFORCE_EQ(axis_names_.find(axis_name) == axis_names_.npos,
                        true,
                        ::common::errors::InvalidArgument(
                            "{} has already existed.", axis_name));
      axis_names_ += axis_name;
    }
    int offset = 'A' - 'a';
    for (auto& axis : axes_) {
      PADDLE_ENFORCE_EQ(
          axis->name.size(),
          1U,
          ::common::errors::InvalidArgument("axis name size must be 1"));
      auto axis_name = axis->name[0];
      if (axis_name >= 'a' && axis_name <= 'z') {
        PADDLE_ENFORCE_NE(
            axis_names_.find(axis_name + offset) == axis_names_.npos,
            true,
            ::common::errors::InvalidArgument(
                "sub-axis {} finds no primal axis", axis_name));
      }
    }
  }
}
Layout::Layout(const std::string& name) {
  PADDLE_ENFORCE_NE(
      name.empty(),
      true,
      ::common::errors::InvalidArgument("The name should not be empty."));
  int factor = 0;
  std::vector<Var> axes;
  for (char c : name) {
    if (c >= 'A' && c <= 'Z') {
      PADDLE_ENFORCE_EQ(factor,
                        0,
                        ::common::errors::InvalidArgument(
                            "The factor should be equal to 0."));
      axes.push_back(ir::Var(std::string(1, c)));
    } else if (c >= '0' && c <= '9') {
      factor = 10 * factor + c - '0';
    } else if (c >= 'a' && c <= 'z') {
      PADDLE_ENFORCE_GT(factor,
                        0,
                        ::common::errors::InvalidArgument(
                            "The factor should be greater than 0."));
      axes.push_back(ir::Var(factor, std::string(1, c)));
      factor = 0;
    } else {
      std::stringstream ss;
      ss << "Invalid layout: " << name;
      PADDLE_THROW(::common::errors::InvalidArgument(ss.str()));
    }
  }
  name_ = name;
  axes_ = axes;
  Verify();
}

}  // namespace ir
}  // namespace cinn
