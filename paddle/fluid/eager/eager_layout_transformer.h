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
#include "paddle/fluid/imperative/layout_autotune.h"
namespace egr {
class EagerLayoutTransformer {
 public:
  EagerLayoutTransformer() : op_name_("") {}
  explicit EagerLayoutTransformer(const std::string& op_name)
      : op_name_(op_name) {
    use_autotune_ = false;
    desired_layout_ = "UNDEFINED";
  }
  template <typename T1>
  void SetAttr(T1 attr) {
    use_autotune_ = false;
    desired_layout_ = "UNDEFINED";
  }
  template <typename T1, typename T2>
  void SetAttr(T1 attr1, T2 attr2) {}

  virtual ~EagerLayoutTransformer() {}

  virtual paddle::experimental::Tensor TransInTensor(
      const paddle::experimental::Tensor in) {
    return in;
  }

 protected:
  bool use_autotune_;
  const std::string op_name_;
  std::string desired_layout_;
};

class EagerHeavilyLayoutSensitiveOpTransformer : public EagerLayoutTransformer {
 public:
  EagerHeavilyLayoutSensitiveOpTransformer() : op_name_("") {}
  explicit EagerHeavilyLayoutSensitiveOpTransformer(const std::string& op_name)
      : op_name_(op_name) {
    use_autotune_ = false;
    final_layout_ = "UNDEFINED";
  }
  template <typename T1>
  void SetAttr(T1 layout) {
    // Step 1: Adjust the data_layout attr to the desired layout
    auto desired_layout =
        paddle::imperative::LayoutAutoTune::Instance().GetDesiredLayout();
    std::string desired_layout_str =
        paddle::framework::DataLayoutToString(desired_layout);
    if (layout != desired_layout_str) {
      VLOG(4) << "Origin layout attr: " << layout
              << ", Desired layout attr: " << desired_layout_str;
      final_layout_ = desired_layout_str;
      use_autotune_ = true;
    }
  }

  virtual paddle::experimental::Tensor TransInTensor(
      paddle::experimental::Tensor in) {
    std::vector<int> axis;
    if (paddle::framework::DataLayoutToString(in.layout()) != final_layout_) {
      if (final_layout_ == "NHWC") {
        axis = {0, 2, 3, 1};
      } else if (final_layout_ == "NCHW") {
        axis = {0, 3, 1, 2};
      } else {
        axis = {0, 1, 2, 3};
      }
      auto out_tensor = transpose_final_state_dygraph_function(in, axis);
      VLOG(4) << "Transpose asdfasdfas ";
      return out_tensor;
    }
    return in;
  }

  std::string GetOutLayout() { return final_layout_; }

  virtual ~EagerHeavilyLayoutSensitiveOpTransformer() {}

 protected:
  bool use_autotune_;
  const std::string op_name_;
  std::string final_layout_;
};

}  // namespace egr
