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
#include "paddle/fluid/eager/eager_layout_transformer.h"
#include "paddle/fluid/imperative/layout_autotune.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
namespace egr {
inline bool NeedTransLayout(
    const paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               kSlotSmallVectorSize>& tensors_vector,
    const paddle::experimental::DataLayout& layout) {
  for (size_t i = 0; i < tensors_vector.size(); i++) {
    for (size_t idx = 0; idx < tensors_vector[0].size(); idx++) {
      if (layout != tensors_vector[i][idx].layout()) {
        return true;
      }
    }
  }
  return false;
}
inline std::shared_ptr<EagerLayoutTransformer> BaseTransformer(
    const std::string& op_name,
    const paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               kSlotSmallVectorSize>& tensors_vector) {
  std::shared_ptr<EagerLayoutTransformer> transposer = nullptr;
  bool unstart =
      (paddle::imperative::LayoutAutoTune::Instance().GetDesiredLayout() ==
       paddle::experimental::DataLayout::UNDEFINED);
  auto first_layout = tensors_vector[0][0].layout();
  VLOG(3) << "Layout autotune was is start ? " << (!unstart) << op_name
          << "'s layout is " << first_layout;

  transposer = std::make_shared<EagerLayoutTransformer>(
      op_name, tensors_vector, first_layout);
  return transposer;
}

// For agnostic op like add, relu, exp
inline std::shared_ptr<EagerLayoutTransformer> EagerLayoutAutotune(
    const std::string& op_name,
    const paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               kSlotSmallVectorSize>& tensors_vector) {
  auto desired_layout =
      paddle::imperative::LayoutAutoTune::Instance().GetDesiredLayout();
  auto default_layout =
      paddle::imperative::LayoutAutoTune::Instance().GetDefaultLayout();
  auto first_layout = tensors_vector[0][0].layout();
  if (NeedTransLayout(tensors_vector, first_layout)) {
    bool need_trans_back = false;
    for (size_t i = 0; i < tensors_vector.size(); i++) {
      for (size_t idx = 0; idx < tensors_vector[0].size(); idx++) {
        if (4 != tensors_vector[i][idx].shape().size()) {
          need_trans_back = true;
          VLOG(3) << "Agnostic op " << op_name << " shape is "
                  << tensors_vector[i][idx].shape().size() << " and layout is "
                  << tensors_vector[i][idx].layout();
        }
      }
    }
    auto final_layout = need_trans_back ? default_layout : desired_layout;
    return std::make_shared<EagerLayoutTransformer>(
        op_name, tensors_vector, final_layout);
  }
  return BaseTransformer(op_name, tensors_vector);
}

// For lightly op like reduce
template <typename T>
inline std::shared_ptr<EagerLayoutTransformer> EagerLayoutAutotune(
    const std::string& op_name,
    const paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               kSlotSmallVectorSize>& tensors_vector,
    T* attr) {
  VLOG(3) << "Lightly op " << op_name << "'s shape is "
          << tensors_vector[0][0].shape().size() << " and layout is "
          << tensors_vector[0][0].layout();

  std::shared_ptr<EagerLayoutTransformer> transposer = nullptr;
  transposer =
      std::make_shared<EagerLightlyLayoutSensitiveOpTransformer>(op_name);
  return transposer;
}

// For lightly op like argmax
template <typename T1, typename T2>
inline std::shared_ptr<EagerLayoutTransformer> EagerLayoutAutotune(
    const std::string& op_name,
    const paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               kSlotSmallVectorSize>& tensors_vector,
    T1* axis,
    T2* keep_dim) {
  VLOG(3) << "Lightly op " << op_name << "'s shape is "
          << tensors_vector[0][0].shape().size() << " and layout is "
          << tensors_vector[0][0].layout();

  return EagerLayoutAutotune<T1>(op_name, tensors_vector, axis);
}

// heavily string data_format, data_layout
template <>
inline std::shared_ptr<EagerLayoutTransformer> EagerLayoutAutotune(
    const std::string& op_name,
    const paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               kSlotSmallVectorSize>& tensors_vector,
    std::string* attr) {
  auto first_layout = tensors_vector[0][0].layout();
  auto transposer = std::make_shared<EagerLayoutTransformer>(
      op_name, tensors_vector, first_layout);
  if (paddle::imperative::LayoutAutoTune::Instance().GetDesiredLayout() ==
      paddle::experimental::DataLayout::UNDEFINED) {
    // Layout autotune only supports model with convolutional layers
    VLOG(3) << "Optimze Layout was not started " << op_name;
    if (op_name != "conv2d") {
      return transposer;
    } else {
      auto data_type = tensors_vector[0][0].dtype();
      bool is_tune_fp32 =
          (data_type == paddle::experimental::DataType::FLOAT32) &&
          (*attr == "NHWC");
      bool is_tune_fp16 =
          (data_type == paddle::experimental::DataType::FLOAT16) &&
          (*attr == "NCHW");
      VLOG(3) << "Conv2d_dy's dtype " << data_type << " format" << (*attr);
      if (is_tune_fp32) {
        paddle::imperative::LayoutAutoTune::Instance().SetDesiredLayout(
            paddle::experimental::DataLayout::NCHW);

        paddle::imperative::LayoutAutoTune::Instance().SetDefaultLayout(
            paddle::experimental::DataLayout::NHWC);
      } else if (is_tune_fp16) {
        paddle::imperative::LayoutAutoTune::Instance().SetDesiredLayout(
            paddle::experimental::DataLayout::NHWC);
        paddle::imperative::LayoutAutoTune::Instance().SetDefaultLayout(
            paddle::experimental::DataLayout::NCHW);
      } else {
        egr::Controller::Instance().DisableLayoutAutoTune();
        return transposer;
      }
      VLOG(3)
          << "Tune the layout from " << *attr << " to "
          << paddle::imperative::LayoutAutoTune::Instance().GetDesiredLayout();
    }
  }

  if (paddle::imperative::LayoutAutoTune::Instance().IsHeavilyLayoutSensitive(
          op_name)) {
    VLOG(3)
        << op_name
        << "'s LayoutTransformer is EagerHeavilyLayoutSensitiveOpTransformer";
    auto heavily_transposer =
        std::make_shared<EagerHeavilyLayoutSensitiveOpTransformer>(op_name,
                                                                   attr);
    return heavily_transposer;
  }

  VLOG(3) << op_name << "'s LayoutTransformer is unimplemented. Use default.";
  return transposer;
}

// lightly  transpose
template <>
inline std::shared_ptr<EagerLayoutTransformer> EagerLayoutAutotune(
    const std::string& op_name,
    const paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               kSlotSmallVectorSize>& tensors_vector,
    std::vector<int>* attr) {
  auto first_layout = tensors_vector[0][0].layout();
  std::shared_ptr<EagerLayoutTransformer> transposer = nullptr;
  if (paddle::imperative::LayoutAutoTune::Instance().GetDesiredLayout() ==
      paddle::experimental::DataLayout::UNDEFINED) {
    VLOG(3) << "Optimze Layout was not started" << op_name;
    transposer = std::make_shared<EagerLayoutTransformer>(
        op_name, tensors_vector, first_layout);
    return transposer;
  }
  if (op_name == "transpose2" &&
      (tensors_vector[0][0].layout() ==
       paddle::imperative::LayoutAutoTune::Instance().GetDesiredLayout())) {
    auto trans = std::make_shared<EagerTransposeOpTransformer>(op_name);
    trans->SetAttr(attr,
                   tensors_vector[0][0].layout() ==
                       paddle::experimental::DataLayout::NHWC);
    return trans;
  }
  transposer =
      std::make_shared<EagerLightlyLayoutSensitiveOpTransformer>(op_name);
  return transposer;
}

// lightly int argmax
template <>
inline std::shared_ptr<EagerLayoutTransformer>
EagerLayoutAutotune<paddle::experimental::Scalar, bool>(
    const std::string& op_name,
    const paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               kSlotSmallVectorSize>& tensors_vector,
    paddle::experimental::Scalar* axis,
    bool* keep_dim) {
  auto first_layout = tensors_vector[0][0].layout();
  std::shared_ptr<EagerLayoutTransformer> transposer = nullptr;
  if (paddle::imperative::LayoutAutoTune::Instance().GetDesiredLayout() ==
      paddle::experimental::DataLayout::UNDEFINED) {
    VLOG(3) << "Optimze Layout was not started" << op_name;
    transposer = std::make_shared<EagerLayoutTransformer>(
        op_name, tensors_vector, first_layout);
    return transposer;
  }
  auto desired_layout =
      paddle::imperative::LayoutAutoTune::Instance().GetDesiredLayout();
  if (op_name == "argmax" &&
      (tensors_vector[0][0].layout() == desired_layout) && (*keep_dim)) {
    std::shared_ptr<EagerArgmaxOpTransformer> argmax_transform = nullptr;
    argmax_transform = std::make_shared<EagerArgmaxOpTransformer>(op_name);
    argmax_transform->SetAttr(axis,
                              tensors_vector[0][0].layout() ==
                                  paddle::experimental::DataLayout::NHWC);
    return argmax_transform;
  }
  transposer =
      std::make_shared<EagerLightlyLayoutSensitiveOpTransformer>(op_name);
  return transposer;
}

// lightly for flatten
template <>
inline std::shared_ptr<EagerLayoutTransformer> EagerLayoutAutotune<int, int>(
    const std::string& op_name,
    const paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               kSlotSmallVectorSize>& tensors_vector,
    int* start_axis,
    int* stop_axis) {
  auto first_layout = tensors_vector[0][0].layout();
  std::shared_ptr<EagerLayoutTransformer> transposer = nullptr;
  auto desired_layout =
      paddle::imperative::LayoutAutoTune::Instance().GetDesiredLayout();
  if (desired_layout == paddle::experimental::DataLayout::UNDEFINED) {
    VLOG(3) << "Optimze Layout was not started" << op_name;
    transposer = std::make_shared<EagerLayoutTransformer>(
        op_name, tensors_vector, first_layout);
    return transposer;
  }
  bool no_tranpose = tensors_vector[0][0].layout() == desired_layout;
  bool is_valid = ((*start_axis) == 1 && (*stop_axis) == 3);
  if (op_name == "flatten" || op_name == "flatten_contiguous_range") {
    if (no_tranpose && is_valid) {
      std::shared_ptr<EagerFlattenOpTransformer> flatten_transform = nullptr;
      flatten_transform = std::make_shared<EagerFlattenOpTransformer>(op_name);
      return flatten_transform;
    }
  }

  transposer =
      std::make_shared<EagerLightlyLayoutSensitiveOpTransformer>(op_name);
  return transposer;
}

// lightly int Concat
template <>
inline std::shared_ptr<EagerLayoutTransformer>
EagerLayoutAutotune<paddle::experimental::Scalar>(
    const std::string& op_name,
    const paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               kSlotSmallVectorSize>& tensors_vector,
    paddle::experimental::Scalar* axis) {
  auto desired_layout =
      paddle::imperative::LayoutAutoTune::Instance().GetDesiredLayout();
  auto first_layout = tensors_vector[0][0].layout();
  std::shared_ptr<EagerLayoutTransformer> transposer = nullptr;
  if (desired_layout == paddle::experimental::DataLayout::UNDEFINED) {
    VLOG(3) << "Optimze Layout was not started" << op_name;
    transposer = std::make_shared<EagerLayoutTransformer>(
        op_name, tensors_vector, first_layout);
    return transposer;
  }

  if (NeedTransLayout(tensors_vector, desired_layout)) {
    VLOG(3) << op_name << " need transpose to default layout";
    transposer =
        std::make_shared<EagerLightlyLayoutSensitiveOpTransformer>(op_name);
    return transposer;
  } else {
    auto trans = std::make_shared<EagerConcatOpTransformer>(op_name);
    trans->SetAttr(axis, desired_layout);
    return trans;
  }
}

}  // namespace egr
