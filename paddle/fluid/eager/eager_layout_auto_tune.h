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
inline LayoutTransformer EagerAutotuneLayoutTransformer(
    const std::string& op_name,
    const paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               kSlotSmallVectorSize>& tensors_vector) {
  VLOG(3) << "asdf Optimze Layout agnostic op: " << op_name;
  auto transposer = LayoutTransformer(op_name);
  return transposer;
}

template <typename T>  // default int
inline LayoutTransformer EagerAutotuneLayoutTransformer(
    const std::string& op_name,
    const paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               kSlotSmallVectorSize>& tensors_vector,
    const T& attr) {
  VLOG(3) << "asdf Optimze Layout agnostic op: " << op_name;
  auto transposer = LayoutTransformer(op_name);
  return transposer;
}
template <>  // default int
inline LayoutTransformer EagerAutotuneLayoutTransformer(
    const std::string& op_name,
    const paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               kSlotSmallVectorSize>& tensors_vector,
    const std::string& attr) {
  auto transposer = LayoutTransformer(op_name);
  if (paddle::imperative::LayoutAutoTune::Instance().GetDesiredLayout() ==
      paddle::experimental::DataLayout::UNDEFINED) {
    // Layout autotune only supports model with convolutional layers
    if (op_name != "conv2d") {
      return transposer;
    } else {
      VLOG(3) << "asdf Optimze Layout lightly or heavily op: " << op_name;
      auto data_type = tensors_vector[0][0].dtype();
      bool is_tune_fp32 =
          (data_type == paddle::experimental::DataType::FLOAT32) &&
          (attr == "NHWC");
      bool is_tune_fp16 =
          (data_type == paddle::experimental::DataType::FLOAT16) &&
          (attr == "NCHW");
      if (is_tune_fp32) {
        paddle::imperative::LayoutAutoTune::Instance().SetDesiredLayout(
            paddle::experimental::DataLayout::NCHW);
      } else if (is_tune_fp16) {
        paddle::imperative::LayoutAutoTune::Instance().SetDesiredLayout(
            paddle::experimental::DataLayout::NHWC);
      } else {
        VLOG(3) << "DisableLayoutAutoTune";
        paddle::imperative::LayoutAutoTune::Instance().DisableLayoutAutoTune();
        return transposer;
      }
      VLOG(3) << "Tune the layout from " << attr << " to "
              << paddle::framework::DataLayoutToString(
                     paddle::imperative::LayoutAutoTune::Instance()
                         .GetDesiredLayout());
    }
  }

  // set layout
  return transposer;
}

template <typename T1, typename T2>  // default int
inline LayoutTransformer EagerAutotuneLayoutTransformer(
    const std::string& op_name,
    const paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               kSlotSmallVectorSize>& amp_tensors_vector,
    const T1& attr1,
    const T2& attr2) {
  VLOG(4) << "asdf Layout asdf AmpAutoCasts: inputs(" << op_name;
  auto transposer = LayoutTransformer(op_name);
  transposer.SetAttr<T1, T2>(attr1, attr2);
  return transposer;
}

}  // namespace egr
