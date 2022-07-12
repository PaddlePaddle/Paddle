// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/eager/api/generated/eager_generated/forwards/dygraph_functions.h"
namespace egr {
class LayoutTransformer {
 public:
  explicit LayoutTransformer(const std::string& op_name) : op_name_(op_name) {
    use_autotune_ = false;
    desired_layout_ = "Undefine";
  }
  template <typename T1>
  void SetAttr(T1 attr) {}
  template <typename T1, typename T2>
  void SetAttr(T1 attr1, T2 attr2) {}

  virtual ~LayoutTransformer() {}

  virtual paddle::experimental::Tensor TransInTensor(
      const paddle::experimental::Tensor in) {
    return in;
  }

 protected:
  bool use_autotune_;
  std::string desired_layout_;

  const std::string op_name_;
  std::vector<std::string> outs_{};
  std::vector<std::string> attrs_{};
};

}  // namespace egr
