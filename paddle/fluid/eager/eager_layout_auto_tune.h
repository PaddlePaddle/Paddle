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
    const paddle::small_vector<std::vector<paddle::Tensor>,
                               kSlotSmallVectorSize>& tensors_vector,
    const phi::DataLayout& layout) {
  for (size_t i = 0; i < tensors_vector.size(); i++) {
    for (size_t idx = 0; idx < tensors_vector[0].size(); idx++) {
      if (layout != tensors_vector[i][idx].layout()) {
        return true;
      }
    }
  }
  return false;
}

inline std::shared_ptr<EagerLayoutTransformer> EagerLayoutAutotune(
    const std::string& op_name,
    const paddle::small_vector<std::vector<paddle::Tensor>,
                               kSlotSmallVectorSize>& tensors_vector) {
  // For agnostic op like add, relu, exp
  auto first_layout = tensors_vector[0][0].layout();
  auto desired_layout = DesiredLayout();
  bool is_started = !(desired_layout == phi::DataLayout::UNDEFINED);
  if (is_started && NeedTransLayout(tensors_vector, first_layout)) {
    bool need_trans_back = false;
    for (size_t i = 0; i < tensors_vector.size(); i++) {
      for (size_t idx = 0; idx < tensors_vector[0].size(); idx++) {
        if (4 != tensors_vector[i][idx].shape().size()) {
          need_trans_back = true;
        }
      }
    }
    auto final_layout = need_trans_back ? DefaultLayout() : desired_layout;
    VLOG(4) << op_name << "'s has different layout, need trans to "
            << final_layout;
    return std::make_shared<EagerLayoutTransformer>(
        op_name, tensors_vector, final_layout);
  }
  return std::make_shared<EagerLayoutTransformer>(
      op_name, tensors_vector, first_layout);
}

template <typename T>
inline std::shared_ptr<EagerLayoutTransformer> EagerLayoutAutotune(
    const std::string& op_name,
    const paddle::small_vector<std::vector<paddle::Tensor>,
                               kSlotSmallVectorSize>& tensors_vector,
    T* attr UNUSED) {
  // For lightly op like reduce
  if ((DesiredLayout() == phi::DataLayout::UNDEFINED)) {
    VLOG(4) << "LayoutAutotune was unstarted. Current op :" << op_name;
    return std::make_shared<EagerLayoutTransformer>(
        op_name, tensors_vector, tensors_vector[0][0].layout());
  }
  return std::make_shared<EagerLightlyLayoutSensitiveOpTransformer>(op_name);
}

template <typename T1, typename T2>
inline std::shared_ptr<EagerLayoutTransformer> EagerLayoutAutotune(
    const std::string& op_name,
    const paddle::small_vector<std::vector<paddle::Tensor>,
                               kSlotSmallVectorSize>& tensors_vector,
    T1* axis,
    T2* keep_dim UNUSED) {
  // For lightly op like argmax
  return EagerLayoutAutotune<T1>(op_name, tensors_vector, axis);
}
template <>
inline std::shared_ptr<EagerLayoutTransformer> EagerLayoutAutotune(
    const std::string& op_name,
    const paddle::small_vector<std::vector<paddle::Tensor>,
                               kSlotSmallVectorSize>& tensors_vector,
    paddle::experimental::IntArray* paddings,
    std::string* attr) {
  // for pad
  if ((DesiredLayout() == phi::DataLayout::UNDEFINED)) {
    VLOG(4) << "LayoutAutotune was unstarted. Current op :" << op_name;
    return std::make_shared<EagerLayoutTransformer>(
        op_name, tensors_vector, tensors_vector[0][0].layout());
  }
  return std::make_shared<EagerLightlyLayoutSensitiveOpTransformer>(op_name);
}
template <>
inline std::shared_ptr<EagerLayoutTransformer> EagerLayoutAutotune(
    const std::string& op_name,
    const paddle::small_vector<std::vector<paddle::Tensor>,
                               kSlotSmallVectorSize>& tensors_vector,
    std::string* attr) {
  // Heavily op with (string) data_format, data_layout
  auto transposer = std::make_shared<EagerLayoutTransformer>(
      op_name, tensors_vector, tensors_vector[0][0].layout());
  if (DesiredLayout() == phi::DataLayout::UNDEFINED) {
    // Layout autotune only supports model with convolutional layers
    if (op_name != "conv2d") {
      VLOG(4) << "LayoutAutotune was unstarted. Current op :" << op_name;
      return transposer;
    } else {
      auto data_type = tensors_vector[0][0].dtype();
      bool is_tune_fp32 =
          (data_type == phi::DataType::FLOAT32) && (*attr == "NHWC");
      bool is_tune_fp16 = (data_type == phi::DataType::FLOAT16 ||
                           data_type == phi::DataType::BFLOAT16) &&
                          (*attr == "NCHW");
      VLOG(4) << "LayoutAutoTune assert with dtype and layout, Current op : "
              << op_name;
      if (is_tune_fp32) {
        paddle::imperative::LayoutAutoTune::Instance().SetDesiredLayout(
            phi::DataLayout::NCHW);

        paddle::imperative::LayoutAutoTune::Instance().SetDefaultLayout(
            phi::DataLayout::NHWC);
      } else if (is_tune_fp16) {
        paddle::imperative::LayoutAutoTune::Instance().SetDesiredLayout(
            phi::DataLayout::NHWC);
        paddle::imperative::LayoutAutoTune::Instance().SetDefaultLayout(
            phi::DataLayout::NCHW);
      } else {
        VLOG(4) << "DisableLayoutAutoTune according to Conv op"
                << " dtype : " << data_type << " format : " << (*attr);
        egr::Controller::Instance().DisableLayoutAutoTune();
        return transposer;
      }
      VLOG(4) << "LayoutAutoTune from " << *attr << " to " << DesiredLayout();
    }
  }

  if (paddle::imperative::LayoutAutoTune::Instance().IsHeavilyLayoutSensitive(
          op_name)) {
    return std::make_shared<EagerHeavilyLayoutSensitiveOpTransformer>(op_name,
                                                                      attr);
  }
  return std::make_shared<EagerLightlyLayoutSensitiveOpTransformer>(op_name);
}

template <>
inline std::shared_ptr<EagerLayoutTransformer> EagerLayoutAutotune(
    const std::string& op_name,
    const paddle::small_vector<std::vector<paddle::Tensor>,
                               kSlotSmallVectorSize>& tensors_vector,
    std::vector<int>* attr) {
  // lightly  transpose
  if (DesiredLayout() == phi::DataLayout::UNDEFINED) {
    VLOG(4) << "LayoutAutotune was unstarted. Current op :" << op_name;
    return std::make_shared<EagerLayoutTransformer>(
        op_name, tensors_vector, tensors_vector[0][0].layout());
  }

  if ((op_name == "transpose2" || op_name == "trans_layout") &&
      (tensors_vector[0][0].layout() == DesiredLayout())) {
    auto trans = std::make_shared<EagerTransposeOpTransformer>(op_name);
    trans->SetAttr(attr,
                   tensors_vector[0][0].layout() == phi::DataLayout::NHWC);
    return trans;
  }
  return std::make_shared<EagerLightlyLayoutSensitiveOpTransformer>(op_name);
}

// lightly int argmax
template <>
inline std::shared_ptr<EagerLayoutTransformer>
EagerLayoutAutotune<paddle::experimental::Scalar, bool>(
    const std::string& op_name,
    const paddle::small_vector<std::vector<paddle::Tensor>,
                               kSlotSmallVectorSize>& tensors_vector,
    paddle::experimental::Scalar* axis,
    bool* keep_dim) {
  if (DesiredLayout() == phi::DataLayout::UNDEFINED) {
    VLOG(4) << "LayoutAutotune was unstarted. Current op :" << op_name;
    return std::make_shared<EagerLayoutTransformer>(
        op_name, tensors_vector, tensors_vector[0][0].layout());
  }

  if (op_name == "argmax" &&
      (tensors_vector[0][0].layout() == DesiredLayout()) && (*keep_dim)) {
    std::shared_ptr<EagerArgmaxOpTransformer> argmax_transform = nullptr;
    argmax_transform = std::make_shared<EagerArgmaxOpTransformer>(op_name);
    argmax_transform->SetAttr(
        axis, tensors_vector[0][0].layout() == phi::DataLayout::NHWC);
    return argmax_transform;
  }
  return std::make_shared<EagerLightlyLayoutSensitiveOpTransformer>(op_name);
}

template <>
inline std::shared_ptr<EagerLayoutTransformer> EagerLayoutAutotune<int, int>(
    const std::string& op_name,
    const paddle::small_vector<std::vector<paddle::Tensor>,
                               kSlotSmallVectorSize>& tensors_vector,
    int* start_axis,
    int* stop_axis) {
  if (DesiredLayout() == phi::DataLayout::UNDEFINED) {
    VLOG(4) << "Optimize Layout was not started" << op_name;
    return std::make_shared<EagerLayoutTransformer>(
        op_name, tensors_vector, tensors_vector[0][0].layout());
  }

  bool no_transpose = tensors_vector[0][0].layout() == DesiredLayout();
  bool is_valid = ((*start_axis) == 1 && (*stop_axis) == 3);
  if (op_name == "flatten" || op_name == "flatten_contiguous_range") {
    if (no_transpose && is_valid) {
      return std::make_shared<EagerFlattenOpTransformer>(op_name);
    }
  }
  return std::make_shared<EagerLightlyLayoutSensitiveOpTransformer>(op_name);
}

template <>
inline std::shared_ptr<EagerLayoutTransformer>
EagerLayoutAutotune<paddle::experimental::Scalar>(
    const std::string& op_name,
    const paddle::small_vector<std::vector<paddle::Tensor>,
                               kSlotSmallVectorSize>& tensors_vector,
    paddle::experimental::Scalar* axis) {
  if (DesiredLayout() == phi::DataLayout::UNDEFINED) {
    VLOG(4) << "Optimize Layout was not started" << op_name;
    return std::make_shared<EagerLayoutTransformer>(
        op_name, tensors_vector, tensors_vector[0][0].layout());
  }

  auto desired_layout = DesiredLayout();
  if (NeedTransLayout(tensors_vector, desired_layout)) {
    VLOG(4) << op_name << "'s has different layout";
    return std::make_shared<EagerLightlyLayoutSensitiveOpTransformer>(op_name);
  }
  if (op_name == "Concat") {
    if (desired_layout == tensors_vector[0][0].layout() &&
        tensors_vector[0][0].shape().size() == 4) {
      auto trans = std::make_shared<EagerConcatOpTransformer>(op_name);
      trans->SetAttr(axis, desired_layout);
      return trans;
    }
  }
  return std::make_shared<EagerLightlyLayoutSensitiveOpTransformer>(op_name);
}

}  // namespace egr
