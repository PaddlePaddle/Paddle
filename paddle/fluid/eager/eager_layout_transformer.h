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
paddle::experimental::Tensor EagerTraceTransposeOp(
    const paddle::experimental::DataLayout layout,
    const paddle::experimental::Tensor& in) {
  std::vector<int> axis;
  if (layout == paddle::experimental::DataLayout::NHWC) {
    axis = {0, 2, 3, 1};
  } else if (layout == paddle::experimental::DataLayout::NCHW) {
    axis = {0, 3, 1, 2};
  } else {
    axis = {0, 1, 2, 3};
  }
  auto out_tensor = transpose_final_state_dygraph_function(in, axis);
  VLOG(4) << "AutoTune Transpose from "
          << paddle::framework::DataLayoutToString(in.layout()) << " to "
          << paddle::framework::DataLayoutToString(layout);
  return out_tensor;
}

class EagerLayoutTransformer {
 public:
  EagerLayoutTransformer() : op_name_("") {}
  explicit EagerLayoutTransformer(const std::string& op_name)
      : op_name_(op_name) {
    use_autotune_ = false;
    final_layout_ = "UNDEFINED";
  }

  EagerLayoutTransformer(const EagerLayoutTransformer&) = delete;

  EagerLayoutTransformer& operator=(const EagerLayoutTransformer&) = delete;

  virtual ~EagerLayoutTransformer() {}

  template <typename T1>
  void SetAttr(T1* attr) {
    use_autotune_ = false;
    final_layout_ = "UNDEFINED";
  }

  template <typename T1, typename T2>
  void SetAttr(T1* attr1, T2* attr2) {
    use_autotune_ = false;
    final_layout_ = "UNDEFINED";
  }

  virtual paddle::experimental::Tensor TransInTensor(
      const std::string& in_name, const paddle::experimental::Tensor& in) {
    VLOG(4) << "input " << in_name << "'s layout is equal to Desired";
    return in;
  }

  virtual void SetOutTensorLayout(paddle::experimental::Tensor* out_tensor) {
    VLOG(4) << op_name_ << " SetOutTensorLayout update from "
            << paddle::framework::DataLayoutToString(out_tensor->layout())
            << " to " << final_layout_;
    // out_tensor->set_layout(paddle::framework::StringToDataLayout(final_layout_));
  }

  virtual void SetOutTensorLayout(
      std::vector<paddle::experimental::Tensor*>* out_tensor) {
    VLOG(4) << op_name_ << " SetOutTensorLayout update from ";
    for (size_t i = 0; i < out_tensor->size(); i++) {
      // (*out_tensor)[i]->set_layout(paddle::framework::StringToDataLayout(final_layout_));
    }
  }

  virtual void SetOutTensorLayout(
      std::vector<paddle::experimental::Tensor>* out_tensor) {
    VLOG(4) << op_name_ << " SetOutTensorLayout update from ";
    for (size_t i = 0; i < out_tensor->size(); i++) {
      // (*out_tensor)[i].set_layout(paddle::framework::StringToDataLayout(final_layout_));
    }
  }

 protected:
  bool use_autotune_;
  std::string op_name_;
  std::string final_layout_;
};

class EagerHeavilyLayoutSensitiveOpTransformer : public EagerLayoutTransformer {
 public:
  EagerHeavilyLayoutSensitiveOpTransformer() {}
  explicit EagerHeavilyLayoutSensitiveOpTransformer(
      const std::string& op_name) {
    VLOG(4) << op_name << " EagerHeavilyLayoutSensitiveOpTransformer ";
    use_autotune_ = false;
    final_layout_ = "UNDEFINED";
  }

  template <typename T1>
  void SetAttr(T1* layout) {
    auto desired_layout =
        paddle::imperative::LayoutAutoTune::Instance().GetDesiredLayout();
    std::string desired_layout_str =
        paddle::framework::DataLayoutToString(desired_layout);
    if (*layout != desired_layout_str) {
      VLOG(4) << " origin layout attr: " << *layout
              << ", Desired layout attr: " << desired_layout_str;
      final_layout_ = desired_layout_str;
      use_autotune_ = true;
      *layout = final_layout_;
    }
  }

  paddle::experimental::Tensor TransInTensor(
      const std::string& in_name, const paddle::experimental::Tensor& in) {
    auto desired_layout =
        paddle::imperative::LayoutAutoTune::Instance().GetDesiredLayout();
    if (heavily_input_.count(in_name) != 0 && in.layout() != desired_layout) {
      VLOG(4) << " EagerHeavilyLayoutSensitiveOpTransformer " << in_name
              << " need transpose from "
              << paddle::framework::DataLayoutToString(in.layout()) << " to "
              << paddle::framework::DataLayoutToString(desired_layout);
      auto out_tensor = EagerTraceTransposeOp(desired_layout, in);
      out_tensor.set_autotune(true);
      out_tensor.set_layout(desired_layout);
      return out_tensor;
    }
    VLOG(4) << " EagerHeavilyLayoutSensitiveOpTransformer " << in_name
            << "'s layout is equal to Desired";
    return in;
  }

  void SetOutTensorLayout(paddle::experimental::Tensor* out_tensor) {
    VLOG(4) << op_name_ << "Heavily SetOutTensorLayout update from "
            << paddle::framework::DataLayoutToString(out_tensor->layout())
            << " to " << final_layout_;
    out_tensor->set_layout(
        paddle::framework::StringToDataLayout(final_layout_));
  }

  void SetOutTensorLayout(
      std::vector<paddle::experimental::Tensor*>* out_tensor) {
    VLOG(4) << op_name_ << "Heavily SetOutTensorLayout update from ";
    for (size_t i = 0; i < out_tensor->size(); i++) {
      (*out_tensor)[i]->set_layout(
          paddle::framework::StringToDataLayout(final_layout_));
    }
  }

  void SetOutTensorLayout(
      std::vector<paddle::experimental::Tensor>* out_tensor) {
    VLOG(4) << op_name_ << "Heavily SetOutTensorLayout update from ";
    for (size_t i = 0; i < out_tensor->size(); i++) {
      (*out_tensor)[i].set_layout(
          paddle::framework::StringToDataLayout(final_layout_));
    }
  }

 protected:
  std::string final_layout_;
  std::unordered_set<std::string> heavily_input_{"x", "y", "input"};
};

}  // namespace egr
