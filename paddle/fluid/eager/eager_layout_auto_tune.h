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
namespace egr {

// layout_agnostic_ops_
inline std::shared_ptr<EagerLayoutTransformer> EagerLayoutAutotune(
    const std::string& op_name,
    const paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               kSlotSmallVectorSize>& tensors_vector) {
  VLOG(3) << " Optimze Layout agnostic op: " << op_name;
  std::shared_ptr<EagerLayoutTransformer> transposer = nullptr;
  transposer = std::make_shared<EagerLayoutTransformer>(op_name);
  transposer->UpdateFinalLayout(tensors_vector);
  return transposer;
}

// lightly T can be int vector<int> vector<int64_t> IntArray
template <typename T>  // default int
inline std::shared_ptr<EagerLayoutTransformer> EagerLayoutAutotune(
    const std::string& op_name,
    const paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               kSlotSmallVectorSize>& tensors_vector,
    T* attr) {
  VLOG(3) << " Optimze Layout lightly op: " << op_name;
  std::shared_ptr<EagerLayoutTransformer> transposer = nullptr;
  transposer = std::make_shared<EagerLayoutTransformer>(op_name);
  return transposer;
}

// lightly int
template <typename T1, typename T2>
inline std::shared_ptr<EagerLayoutTransformer> EagerLayoutAutotune(
    const std::string& op_name,
    const paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               kSlotSmallVectorSize>& amp_tensors_vector,
    T1* attr1,
    T2* attr2) {
  VLOG(3) << " Optimze Layout lightly op: " << op_name;
  std::shared_ptr<EagerLayoutTransformer> transposer = nullptr;
  transposer = std::make_shared<EagerLayoutTransformer>(op_name);
  return transposer;
}

// heavily
template <>
inline std::shared_ptr<EagerLayoutTransformer> EagerLayoutAutotune(
    const std::string& op_name,
    const paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               kSlotSmallVectorSize>& tensors_vector,
    std::string* attr) {
  auto transposer = std::make_shared<EagerLayoutTransformer>(op_name);
  if (paddle::imperative::LayoutAutoTune::Instance().GetDesiredLayout() ==
      paddle::experimental::DataLayout::UNDEFINED) {
    // Layout autotune only supports model with convolutional layers
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
      if (is_tune_fp32) {
        paddle::imperative::LayoutAutoTune::Instance().SetDesiredLayout(
            paddle::experimental::DataLayout::NCHW);
      } else if (is_tune_fp16) {
        paddle::imperative::LayoutAutoTune::Instance().SetDesiredLayout(
            paddle::experimental::DataLayout::NHWC);
      } else {
        paddle::imperative::LayoutAutoTune::Instance().DisableLayoutAutoTune();
        return transposer;
      }
      VLOG(3) << "Tune the layout from " << attr << " to "
              << paddle::framework::DataLayoutToString(
                     paddle::imperative::LayoutAutoTune::Instance()
                         .GetDesiredLayout());
    }
  }

  if (paddle::imperative::LayoutAutoTune::Instance().IsHeavilyLayoutSensitive(
          op_name)) {
    auto heavily_transposer =
        std::make_shared<EagerHeavilyLayoutSensitiveOpTransformer>(op_name);

    heavily_transposer->SetAttr<std::string>(attr);
    return heavily_transposer;
  } else {
    VLOG(4) << op_name
            << "'s LayoutTransformer is unimplemented. Use default "
               "LayoutTransformer instead.";
    return transposer;
  }

  // set layout
  return transposer;
}

}  // namespace egr
