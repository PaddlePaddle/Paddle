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
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/tensor_utils.h"
namespace egr {
inline paddle::experimental::Tensor EagerTraceTransposeOp(
    const paddle::experimental::DataLayout layout,
    const paddle::experimental::Tensor& in) {
  if (in.shape().size() != 4) {
    VLOG(4) << "Shape is " << in.shape().size() << " can't transpose to"
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
  auto out_tensor = transpose_ad_func(in, axis);
  VLOG(4) << "AutoTune Transpose from "
          << paddle::framework::DataLayoutToString(in.layout()) << " to "
          << paddle::framework::DataLayoutToString(layout);
  return out_tensor;
}

// agnostic op
class EagerLayoutTransformer {
 public:
  EagerLayoutTransformer() : op_name_("") {}
  explicit EagerLayoutTransformer(
      const std::string& op_name,
      const paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                 kSlotSmallVectorSize>& tensors_vector)
      : op_name_(op_name) {
    final_layout_ = "UNDEFINED";
    auto desired_layout =
        paddle::imperative::LayoutAutoTune::Instance().GetDesiredLayout();
    for (size_t i = 0; i < tensors_vector.size(); i++) {
      for (size_t idx = 0; idx < tensors_vector[0].size(); idx++) {
        if (final_layout_ == "UNDEFINED") {
          final_layout_ = paddle::framework::DataLayoutToString(
              tensors_vector[0][0].layout());
        } else if (tensors_vector[i][idx].layout() == desired_layout) {
          final_layout_ = paddle::framework::DataLayoutToString(desired_layout);
          break;
        }
      }
    }
    VLOG(4) << op_name_ << "final_layout_ is  " << final_layout_;
  }

  EagerLayoutTransformer(const EagerLayoutTransformer&) = delete;

  EagerLayoutTransformer& operator=(const EagerLayoutTransformer&) = delete;

  virtual ~EagerLayoutTransformer() {}

  virtual paddle::optional<paddle::experimental::Tensor> TransInTensor(
      const std::string& in_name,
      const paddle::optional<paddle::experimental::Tensor>& in) {
    VLOG(4) << op_name_ << "is is agnostic, final_layout_ is " << final_layout_;
    return in;
  }

  virtual paddle::optional<std::vector<paddle::experimental::Tensor>>
  TransInTensor(
      const std::string& in_name,
      const paddle::optional<std::vector<paddle::experimental::Tensor>>& in) {
    return in;
  }

  virtual std::vector<paddle::experimental::Tensor> TransInTensor(
      const std::string& in_name,
      const std::vector<paddle::experimental::Tensor>& in) {
    return in;
  }

  virtual paddle::experimental::Tensor TransInTensor(
      const std::string& in_name, const paddle::experimental::Tensor& in) {
    return in;
  }

  virtual void SetOutTensorLayout(paddle::experimental::Tensor* out_tensor) {
    bool use_default = (final_layout_ == "Undefined(AnyLayout)" ||
                        final_layout_ == ("UNDEFINED"));
    auto layout = paddle::framework::StringToDataLayout(final_layout_);
    if (!use_default) {
      phi::DenseTensorUtils::GetMutableMeta(
          static_cast<phi::DenseTensor*>(out_tensor->impl().get()))
          ->layout = layout;
    }
    VLOG(4) << op_name_ << "is is agnostic, use_default " << use_default;
  }

  virtual void SetOutTensorLayout(
      std::vector<paddle::experimental::Tensor>* out_tensor) {
    bool use_default = (final_layout_ == "Undefined(AnyLayout)" ||
                        final_layout_ == ("UNDEFINED"));
    if (!use_default) {
      for (size_t i = 0; i < out_tensor->size(); i++) {
        phi::DenseTensorUtils::GetMutableMeta(
            static_cast<phi::DenseTensor*>((*out_tensor)[i].impl().get()))
            ->layout =
            paddle::imperative::LayoutAutoTune::Instance().GetDesiredLayout();
      }
    }
    VLOG(4) << op_name_ << "is is agnostic, use_default " << use_default;
  }

 protected:
  std::string op_name_;
  std::string final_layout_;
};

class EagerHeavilyLayoutSensitiveOpTransformer : public EagerLayoutTransformer {
 public:
  explicit EagerHeavilyLayoutSensitiveOpTransformer(const std::string& op_name,
                                                    std::string* layout)
      : op_name_(op_name),
        desired_layout_(
            paddle::imperative::LayoutAutoTune::Instance().GetDesiredLayout()) {
    VLOG(3) << "Optimze Layout heavily op: " << op_name;
    final_layout_ = paddle::framework::DataLayoutToString(desired_layout_);
    if ((*layout) != final_layout_) {
      *layout = final_layout_;
    }
  }

  virtual paddle::optional<std::vector<paddle::experimental::Tensor>>
  TransInTensor(
      const std::string& in_name,
      const paddle::optional<std::vector<paddle::experimental::Tensor>>& in) {
    VLOG(4) << op_name_ << "is is heavily";
    return in;
  }

  virtual paddle::optional<paddle::experimental::Tensor> TransInTensor(
      const std::string& in_name,
      const paddle::optional<paddle::experimental::Tensor>& in) {
    VLOG(4) << op_name_ << "is is heavily";
    return in;
  }

  paddle::experimental::Tensor TransInTensor(
      const std::string& in_name, const paddle::experimental::Tensor& in) {
    if (heavily_input_.count(in_name) != 0 && in.layout() != desired_layout_) {
      VLOG(4) << op_name_ << "'s " << in_name << " need transpose from "
              << paddle::framework::DataLayoutToString(in.layout()) << " to "
              << final_layout_;
      auto out_tensor = EagerTraceTransposeOp(desired_layout_, in);
      return out_tensor;
    }
    return in;
  }

  void SetOutTensorLayout(paddle::experimental::Tensor* out_tensor) {
    if (out_tensor->layout() != desired_layout_) {
      VLOG(4) << " Set Out_tensor's layout from "
              << paddle::framework::DataLayoutToString(out_tensor->layout())
              << " to " << final_layout_;
      phi::DenseTensorUtils::GetMutableMeta(
          static_cast<phi::DenseTensor*>(out_tensor->impl().get()))
          ->layout = desired_layout_;
    }
  }

  void SetOutTensorLayout(
      std::vector<paddle::experimental::Tensor*>* out_tensor) {
    for (size_t i = 0; i < out_tensor->size(); i++) {
      SetOutTensorLayout((*out_tensor)[i]);
    }
  }

  void SetOutTensorLayout(
      std::vector<paddle::experimental::Tensor>* out_tensor) {
    for (size_t i = 0; i < out_tensor->size(); i++) {
      if ((*out_tensor)[i].layout() != desired_layout_) {
        VLOG(4) << " Set Out_tensor's layout from "
                << paddle::framework::DataLayoutToString(
                       (*out_tensor)[i].layout())
                << " to " << final_layout_;
        phi::DenseTensorUtils::GetMutableMeta(
            static_cast<phi::DenseTensor*>((*out_tensor)[i].impl().get()))
            ->layout = desired_layout_;
      }
    }
  }

 protected:
  std::string op_name_;
  std::string final_layout_;
  const paddle::experimental::DataLayout desired_layout_;
  std::unordered_set<std::string> heavily_input_{"x", "y", "input"};
};

class EagerLightlyLayoutSensitiveOpTransformer : public EagerLayoutTransformer {
 public:
  EagerLightlyLayoutSensitiveOpTransformer() {}
  explicit EagerLightlyLayoutSensitiveOpTransformer(const std::string& op_name)
      : op_name_(op_name) {
    VLOG(3) << "Optimze Layout lightly " << op_name;
    auto desired_layout =
        paddle::imperative::LayoutAutoTune::Instance().GetDesiredLayout();
    final_layout_ = paddle::framework::DataLayoutToString(desired_layout);
  }

  // transpose from desired to default
  paddle::experimental::Tensor TransInTensor(
      const std::string& in_name, const paddle::experimental::Tensor& in) {
    std::string input_layout =
        paddle::framework::DataLayoutToString(in.layout());
    auto default_layout =
        paddle::imperative::LayoutAutoTune::Instance().GetDefaultLayout();

    if (final_layout_ == input_layout && in.shape().size() == 4) {
      VLOG(4) << op_name_ << "'s " << in_name << " need transpose from "
              << input_layout << " to default_layout";
      auto out_tensor = EagerTraceTransposeOp(
          paddle::experimental::DataLayout::UNDEFINED, in);
      phi::DenseTensorUtils::GetMutableMeta(
          static_cast<phi::DenseTensor*>(out_tensor.impl().get()))
          ->layout = default_layout;
      return out_tensor;
    }
    VLOG(4) << in_name << "'s layout is " << input_layout;
    return in;
  }

  virtual std::vector<paddle::experimental::Tensor> TransInTensor(
      const std::string& in_name,
      const std::vector<paddle::experimental::Tensor>& in) {
    std::vector<paddle::experimental::Tensor> result;
    auto desired_layout =
        paddle::imperative::LayoutAutoTune::Instance().GetDesiredLayout();
    auto default_layout =
        paddle::imperative::LayoutAutoTune::Instance().GetDefaultLayout();
    for (size_t i = 0; i < in.size(); i++) {
      auto in_tensor = in[i];
      if (in_tensor.layout() == desired_layout) {
        VLOG(4) << op_name_ << "'s " << in_name << " need transpose from "
                << final_layout_ << " to default_layout";
        auto out_tensor = EagerTraceTransposeOp(
            paddle::experimental::DataLayout::UNDEFINED, in_tensor);
        phi::DenseTensorUtils::GetMutableMeta(
            static_cast<phi::DenseTensor*>(out_tensor.impl().get()))
            ->layout = default_layout;
        result.emplace_back(out_tensor);
      } else {
        result.emplace_back(in_tensor);
      }
    }
    return result;
  }

  void SetOutTensorLayout(paddle::experimental::Tensor* out_tensor) {
    auto out_layout = out_tensor->layout();
    auto default_layout =
        paddle::imperative::LayoutAutoTune::Instance().GetDefaultLayout();
    if (out_layout != default_layout) {
      VLOG(4) << op_name_ << "'s out need transpose to default_layout";
      phi::DenseTensorUtils::GetMutableMeta(
          static_cast<phi::DenseTensor*>(out_tensor->impl().get()))
          ->layout = default_layout;
    }
  }

  void SetOutTensorLayout(
      std::vector<paddle::experimental::Tensor*>* out_tensor) {
    for (size_t i = 0; i < out_tensor->size(); i++) {
      VLOG(4) << "out layout is"
              << paddle::framework::DataLayoutToString(
                     (*out_tensor)[i]->layout());
      SetOutTensorLayout((*out_tensor)[i]);
    }
  }

  void SetOutTensorLayout(
      std::vector<paddle::experimental::Tensor>* out_tensor) {
    auto default_layout =
        paddle::imperative::LayoutAutoTune::Instance().GetDefaultLayout();
    for (size_t i = 0; i < out_tensor->size(); i++) {
      VLOG(4) << " out_tensor layout trans to default ";
      phi::DenseTensorUtils::GetMutableMeta(
          static_cast<phi::DenseTensor*>((*out_tensor)[i].impl().get()))
          ->layout = default_layout;
    }
  }

 protected:
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
    VLOG(3) << "Optimze Layout TransposeOpTransformer " << op_name;
    auto desired_layout =
        paddle::imperative::LayoutAutoTune::Instance().GetDesiredLayout();
    std::string desired_layout_str =
        paddle::framework::DataLayoutToString(desired_layout);
    final_layout_ = desired_layout_str;
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

  paddle::experimental::Tensor TransInTensor(
      const std::string& in_name, const paddle::experimental::Tensor& in) {
    VLOG(4) << "with no transpose: EagerTransposeOpTransformer " << in_name
            << "'s layout is "
            << paddle::framework::DataLayoutToString(in.layout());
    return in;
  }

  void SetOutTensorLayout(paddle::experimental::Tensor* out_tensor) {
    auto desired_layout =
        paddle::imperative::LayoutAutoTune::Instance().GetDesiredLayout();
    if (out_tensor->layout() != desired_layout) {
      VLOG(4) << " Set Out_tensor's layout from "
              << paddle::framework::DataLayoutToString(out_tensor->layout())
              << " to " << final_layout_;
      phi::DenseTensorUtils::GetMutableMeta(
          static_cast<phi::DenseTensor*>(out_tensor->impl().get()))
          ->layout = desired_layout;
    }
  }

 protected:
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
    VLOG(3) << "Optimze Layout lightly " << op_name;
  }

  void SetAttr(paddle::experimental::Scalar* axis, bool is_nhwc) {
    std::vector<int> perm_nhwc = {0, 3, 1, 2};
    std::vector<int> perm_nchw = {0, 2, 3, 1};
    auto perm = is_nhwc ? perm_nhwc : perm_nchw;
    int axes = axis->to<int>();
    (*axis) = static_cast<paddle::experimental::Scalar>(perm[axes]);
  }

  void SetOutTensorLayout(paddle::experimental::Tensor* out_tensor) {
    VLOG(4) << "EagerArgmaxOpTransformer's out layout is"
            << paddle::framework::DataLayoutToString(out_tensor->layout());
    auto desired_layout =
        paddle::imperative::LayoutAutoTune::Instance().GetDesiredLayout();
    if (desired_layout != out_tensor->layout()) {
      VLOG(4) << "Change layout from "
              << paddle::framework::DataLayoutToString(out_tensor->layout())
              << " to " << final_layout_;
      phi::DenseTensorUtils::GetMutableMeta(
          static_cast<phi::DenseTensor*>(out_tensor->impl().get()))
          ->layout = desired_layout;
    }
  }

 protected:
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
    VLOG(3) << "Optimze Layout lightly " << op_name;
    auto desired_layout =
        paddle::imperative::LayoutAutoTune::Instance().GetDesiredLayout();
    std::string desired_layout_str =
        paddle::framework::DataLayoutToString(desired_layout);
    final_layout_ = desired_layout_str;
  }

  // transpose from NHWC to NCHW
  paddle::experimental::Tensor TransInTensor(
      const std::string& in_name, const paddle::experimental::Tensor& in) {
    return in;
  }

  void SetOutTensorLayout(paddle::experimental::Tensor* out_tensor) {
    VLOG(4) << "EagerArgmaxOpTransformer's out layout is"
            << paddle::framework::DataLayoutToString(out_tensor->layout());
    auto layout = paddle::framework::StringToDataLayout(final_layout_);
    if (layout != out_tensor->layout()) {
      VLOG(4) << "Change layout from "
              << paddle::framework::DataLayoutToString(out_tensor->layout())
              << " to " << final_layout_;
      phi::DenseTensorUtils::GetMutableMeta(
          static_cast<phi::DenseTensor*>(out_tensor->impl().get()))
          ->layout = layout;
    }
  }

 protected:
  std::string op_name_;
  std::string final_layout_;
  std::unordered_set<std::string> heavily_input_{"x", "y", "input"};
};

class EagerConcatOpTransformer
    : public EagerLightlyLayoutSensitiveOpTransformer {
 public:
  EagerConcatOpTransformer() {}
  explicit EagerConcatOpTransformer(const std::string& op_name)
      : op_name_(op_name) {
    VLOG(3) << "Optimze Layout lightly " << op_name;
    auto desired_layout =
        paddle::imperative::LayoutAutoTune::Instance().GetDesiredLayout();
    std::string desired_layout_str =
        paddle::framework::DataLayoutToString(desired_layout);
    final_layout_ = desired_layout_str;
  }

  void SetAttr(paddle::experimental::Scalar* axis,
               paddle::framework::DataLayout layout) {
    std::vector<int> perm_nhwc = {0, 3, 1, 2};
    std::vector<int> perm_nchw = {0, 2, 3, 1};
    int axes = axis->to<int>();
    auto perm =
        (paddle::framework::DataLayout::NHWC == layout) ? perm_nhwc : perm_nchw;
    (*axis) = static_cast<paddle::experimental::Scalar>(perm[axes]);
  }

  virtual std::vector<paddle::experimental::Tensor> TransInTensor(
      const std::string& in_name,
      const std::vector<paddle::experimental::Tensor>& in) {
    return in;
  }

  void SetOutTensorLayout(paddle::experimental::Tensor* out_tensor) {
    auto layout = paddle::framework::StringToDataLayout(final_layout_);
    if (layout != out_tensor->layout()) {
      VLOG(4) << "Change layout from "
              << paddle::framework::DataLayoutToString(out_tensor->layout())
              << " to " << final_layout_;
      phi::DenseTensorUtils::GetMutableMeta(
          static_cast<phi::DenseTensor*>(out_tensor->impl().get()))
          ->layout = layout;
    }
  }

 protected:
  std::string op_name_;
  std::string final_layout_;
  std::unordered_set<std::string> heavily_input_{"x", "y", "input"};
};
}  // namespace egr
