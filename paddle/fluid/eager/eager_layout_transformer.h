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
inline paddle::experimental::Tensor EagerTraceTransposeOp(
    const paddle::experimental::DataLayout layout,
    const paddle::experimental::Tensor& in) {
  if (in.shape().size() != 4) {
    VLOG(4) << "Tensor's shape is " << in.shape().size()
            << " can't transpose to"
            << paddle::framework::DataLayoutToString(layout);
    return in;
  }
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

  virtual paddle::experimental::Tensor TransInTensor(
      const std::string& in_name, const paddle::experimental::Tensor& in) {
    VLOG(4) << "TransInTensor " << in_name << "'s layout is "
            << paddle::framework::DataLayoutToString(in.layout())
            << " and final_layout_  is " << final_layout_;
    if (final_layout_ == "UNDEFINED") {
      final_layout_ = paddle::framework::DataLayoutToString(in.layout());
    }
    VLOG(4) << "TransInTensor final_layout_ is " << final_layout_;
    return in;
  }

  virtual void SetOutTensorLayout(paddle::experimental::Tensor* out_tensor) {
    if (final_layout_ != "Undefined(AnyLayout)" &&
        final_layout_ != "UNDEFINED") {
      VLOG(4) << op_name_ << " SetOutTensorLayout update from "
              << paddle::framework::DataLayoutToString(out_tensor->layout())
              << " to " << final_layout_;
      out_tensor->set_layout(
          paddle::framework::StringToDataLayout(final_layout_));
    } else {
      VLOG(4) << op_name_ << " SetOutTensorLayout final_layout_ is  "
              << final_layout_ << " so out_tensor's layout will be default";
    }
  }

  virtual void SetOutTensorLayout(
      std::vector<paddle::experimental::Tensor*>* out_tensor) {
    if (final_layout_ != "Undefined(AnyLayout)" ||
        final_layout_ != "UNDEFINED") {
      for (size_t i = 0; i < out_tensor->size(); i++) {
        VLOG(4) << op_name_ << " SetOutTensorLayout to " << final_layout_;
        (*out_tensor)[i]->set_layout(
            paddle::framework::StringToDataLayout(final_layout_));
      }
    } else {
      VLOG(4) << op_name_ << " SetOutTensorLayout final_layout_ is  "
              << final_layout_ << " so out_tensor's layout will be default";
    }
  }

  virtual void SetOutTensorLayout(
      std::vector<paddle::experimental::Tensor>* out_tensor) {
    if (final_layout_ != "Undefined(AnyLayout)" ||
        final_layout_ != "UNDEFINED") {
      for (size_t i = 0; i < out_tensor->size(); i++) {
        VLOG(4) << op_name_ << " SetOutTensorLayout to " << final_layout_;
        (*out_tensor)[i].set_layout(
            paddle::framework::StringToDataLayout(final_layout_));
      }
    } else {
      VLOG(4) << op_name_ << " SetOutTensorLayout final_layout_ is  "
              << final_layout_ << " so out_tensor's layout will be default";
    }
  }

  void UpdateFinalLayout(
      const paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                 kSlotSmallVectorSize>& tensors_vector) {
    auto desired_layout =
        paddle::imperative::LayoutAutoTune::Instance().GetDesiredLayout();
    for (size_t i = 0; i < tensors_vector.size(); i++) {
      for (size_t idx = 0; idx < tensors_vector[0].size(); idx++) {
        VLOG(4) << op_name_ << " UpdateFinalLayout final_layout_ is  "
                << final_layout_ << " tensors_vector[" << i << "][" << idx
                << "]'s layout is "
                << paddle::framework::DataLayoutToString(
                       tensors_vector[i][idx].layout());
        final_layout_ = paddle::framework::DataLayoutToString(
            tensors_vector[0][0].layout());
        if (tensors_vector[i][idx].layout() == desired_layout) {
          final_layout_ = paddle::framework::DataLayoutToString(desired_layout);
          break;
        }
      }
    }

    VLOG(4) << op_name_ << " UpdateFinalLayout final_layout_ is  "
            << final_layout_;
  }

 protected:
  bool use_autotune_;
  std::string op_name_;
  std::string final_layout_;
};

class EagerHeavilyLayoutSensitiveOpTransformer : public EagerLayoutTransformer {
 public:
  EagerHeavilyLayoutSensitiveOpTransformer() {}
  explicit EagerHeavilyLayoutSensitiveOpTransformer(const std::string& op_name)
      : op_name_(op_name) {
    VLOG(4) << op_name << " EagerHeavilyLayoutSensitiveOpTransformer ";
    use_autotune_ = false;
    final_layout_ = "NCHW";
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
      VLOG(4) << op_name_ << " EagerHeavilyLayoutSensitiveOpTransformer "
              << in_name << " need transpose from "
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
  bool use_autotune_;
  std::string op_name_;
  std::string final_layout_;
  std::unordered_set<std::string> heavily_input_{"x", "y", "input"};
};

class EagerLightlyLayoutSensitiveOpTransformer : public EagerLayoutTransformer {
 public:
  EagerLightlyLayoutSensitiveOpTransformer() {}
  explicit EagerLightlyLayoutSensitiveOpTransformer(const std::string& op_name)
      : op_name_(op_name) {
    VLOG(4) << op_name << " EagerHeavilyLayoutSensitiveOpTransformer ";
    use_autotune_ = false;
    final_layout_ = "NCHW";
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

  // transpose from NHWC to NCHW
  paddle::experimental::Tensor TransInTensor(
      const std::string& in_name, const paddle::experimental::Tensor& in) {
    auto desired_layout =
        paddle::imperative::LayoutAutoTune::Instance().GetDesiredLayout();
    if (desired_layout == in.layout() && in.shape().size() == 4) {
      VLOG(4) << op_name_ << " EagerLayoutTransformer's " << in_name
              << " need transpose from "
              << paddle::framework::DataLayoutToString(in.layout())
              << " to  NCHW";
      auto out_tensor = EagerTraceTransposeOp(
          paddle::experimental::DataLayout::UNDEFINED, in);
      out_tensor.set_autotune(false);
      out_tensor.set_layout(paddle::experimental::DataLayout::NCHW);
      return out_tensor;
    }
    VLOG(4) << " EagerLightlyLayoutSensitiveOpTransformer's " << in_name
            << "'s layout is equal to Desired and in's shape is "
            << in.shape().size();
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
  bool use_autotune_;
  std::string op_name_;
  std::string final_layout_;
  std::unordered_set<std::string> heavily_input_{"x", "y", "input"};
};

class EagerTransposeOpTransformer
    : public EagerLightlyLayoutSensitiveOpTransformer {
 public:
  EagerTransposeOpTransformer() {}
  explicit EagerTransposeOpTransformer(const std::string& op_name)
      : op_name_(op_name) {
    VLOG(4) << op_name << " is EagerTransposeOpTransformer";
    use_autotune_ = false;
    final_layout_ = "NCHW";
  }

  void SetAttr(std::vector<int>* axis, bool is_nhwc) {
    // input's layout is nhwc and input's layout === desired_layout
    std::vector<int> perm_nchw = {0, 2, 3, 1};
    std::vector<int> perm_nhwc = {0, 3, 1, 2};
    auto perm = is_nhwc ? perm_nhwc : perm_nchw;
    (*axis)[0] = perm[(*axis)[0]];
    (*axis)[1] = perm[(*axis)[1]];
    (*axis)[2] = perm[(*axis)[2]];
    (*axis)[3] = perm[(*axis)[3]];
    VLOG(4) << " EagerTransposeOpTransformer " << op_name_
            << "'s layout is equal to desire: " << is_nhwc;
  }

  // transpose from NHWC to NCHW
  paddle::experimental::Tensor TransInTensor(
      const std::string& in_name, const paddle::experimental::Tensor& in) {
    VLOG(4) << " EagerTransposeOpTransformer " << in_name << "'s layout is "
            << paddle::framework::DataLayoutToString(in.layout());
    return in;
  }

  void SetOutTensorLayout(paddle::experimental::Tensor* out_tensor) {
    VLOG(4) << op_name_
            << " EagerTransposeOpTransformer SetOutTensorLayout update from "
            << paddle::framework::DataLayoutToString(out_tensor->layout())
            << " to " << final_layout_;
    out_tensor->set_layout(
        paddle::framework::StringToDataLayout(final_layout_));
  }

 protected:
  bool use_autotune_;
  std::string op_name_;
  std::string final_layout_;
  std::unordered_set<std::string> heavily_input_{"x", "y", "input"};
};

class EagerArgmaxOpTransformer
    : public EagerLightlyLayoutSensitiveOpTransformer {
 public:
  EagerArgmaxOpTransformer() {}
  explicit EagerArgmaxOpTransformer(const std::string& op_name)
      : op_name_(op_name) {
    VLOG(4) << op_name << " is EagerArgmaxOpTransformer";
    use_autotune_ = false;
    final_layout_ = "UNDEFINED";
  }
  void SetAttr(int64_t* axis, bool* keep_dims, bool is_nhwc) {
    if ((*keep_dims)) {
      std::vector<int> perm_nchw = {0, 2, 3, 1};
      std::vector<int> perm_nhwc = {0, 3, 1, 2};
      auto perm = is_nhwc ? perm_nhwc : perm_nchw;
      (*axis) = perm[(*axis)];
      VLOG(4) << " EagerTransposeOpTransformer " << op_name_
              << "'s layout is equal to desire: " << is_nhwc;
      final_layout_ = "UNDEFINED";
      use_autotune_ = true;
    } else {
      final_layout_ = "UNDEFINED";
      use_autotune_ = false;
    }
  }

  // transpose from NHWC to NCHW
  paddle::experimental::Tensor TransInTensor(
      const std::string& in_name, const paddle::experimental::Tensor& in) {
    if (use_autotune_) {
      final_layout_ = paddle::framework::DataLayoutToString(in.layout());
    }
    VLOG(4) << " EagerTransposeOpTransformer " << in_name << "'s layout is "
            << paddle::framework::DataLayoutToString(in.layout());
    return in;
  }

  void SetOutTensorLayout(paddle::experimental::Tensor* out_tensor) {
    VLOG(4) << op_name_
            << " EagerTransposeOpTransformer SetOutTensorLayout update from "
            << paddle::framework::DataLayoutToString(out_tensor->layout())
            << " to " << final_layout_;
    if (use_autotune_)
      out_tensor->set_layout(
          paddle::framework::StringToDataLayout(final_layout_));
  }

 protected:
  bool use_autotune_;
  std::string op_name_;
  std::string final_layout_;
  std::unordered_set<std::string> heavily_input_{"x", "y", "input"};
};

class EagerFlattenOpTransformer
    : public EagerLightlyLayoutSensitiveOpTransformer {
 public:
  EagerFlattenOpTransformer() {}
  explicit EagerFlattenOpTransformer(const std::string& op_name)
      : op_name_(op_name) {
    VLOG(4) << op_name << " is EagerFlattenOpTransformer";
    use_autotune_ = false;
    final_layout_ = "UNDEFINED";
  }

  void SetAttr(int* start_axis, int* stop_axis, bool is_nhwc) {
    if ((*start_axis) == 1 && (*stop_axis) == 3) {  // input == desire
      final_layout_ = "UNDEFINED";
      use_autotune_ = true;
    } else {
      final_layout_ = "UNDEFINED";
      use_autotune_ = false;
    }
  }

  // transpose from NHWC to NCHW
  paddle::experimental::Tensor TransInTensor(
      const std::string& in_name, const paddle::experimental::Tensor& in) {
    auto desired_layout =
        paddle::imperative::LayoutAutoTune::Instance().GetDesiredLayout();
    final_layout_ = paddle::framework::DataLayoutToString(in.layout());
    if (desired_layout == in.layout() && (!use_autotune_)) {
      VLOG(4) << op_name_ << " EagerLayoutTransformer's " << in_name
              << " need transpose from "
              << paddle::framework::DataLayoutToString(in.layout())
              << " to  NCHW";
      auto out_tensor = EagerTraceTransposeOp(
          paddle::experimental::DataLayout::UNDEFINED, in);
      out_tensor.set_autotune(false);
      out_tensor.set_layout(paddle::experimental::DataLayout::NCHW);
      return out_tensor;
    }
    VLOG(4) << " EagerLightlyLayoutSensitiveOpTransformer's " << in_name
            << "'s layout is not equal to Desired";
    return in;
  }

  void SetOutTensorLayout(paddle::experimental::Tensor* out_tensor) {
    VLOG(4) << op_name_ << " EagerFlattenOpTransformer SetOutTensorLayout is "
            << paddle::framework::DataLayoutToString(out_tensor->layout());
    // if (use_autotune_)
    // out_tensor->set_layout(
    //     paddle::framework::StringToDataLayout(final_layout_));
  }

 protected:
  bool use_autotune_;
  std::string op_name_;
  std::string final_layout_;
  std::unordered_set<std::string> heavily_input_{"x", "y", "input"};
};

}  // namespace egr
