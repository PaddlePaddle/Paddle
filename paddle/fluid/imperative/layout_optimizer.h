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
#include "paddle/fluid/imperative/layout_autotune.h"
#include "paddle/fluid/imperative/layout_transposer.h"

namespace phi {
namespace autotune {

template <typename VarType>
paddle::imperative::NameVarMap<VarType> LayoutOptimizer(
    const std::string& op_type,
    const paddle::imperative::NameVarMap<VarType>& ins,
    const paddle::imperative::NameVarMap<VarType>& outs,
    paddle::framework::AttributeMap* attrs,
    const std::shared_ptr<paddle::imperative::Tracer>& tracer) {
  if (!LayoutAutoTune::Instance().UseLayoutAutoTune()) {
    return ins;
  }

  // When layout autotuning is enabled, the tuner will check the desired layout.
  // (1) If the desired layout is undefined, and there is no convolutional
  // layers, layout optimization is unnecessary. Otherwise, the desired layout
  // will be set to the best layout only when these is a convolutional layer
  // with
  // NCHW-Layout and the TensorCore is available.
  // (2) If the desired layout is defined, run the transposer.

  if (LayoutAutoTune::Instance().GetDesiredLayout() == DataLayout::UNDEFINED) {
    // Layout autotune only supports model with convolutional layers
    if (op_type != "conv2d") {
      return ins;
    } else {
      if (BOOST_GET_CONST(std::string, (*attrs)["data_format"]) == "NCHW") {
        LayoutAutoTune::Instance().SetDesiredLayout(DataLayout::NHWC);
        VLOG(3) << "Tune the layout from "
                << BOOST_GET_CONST(std::string, (*attrs)["data_format"])
                << " to " << paddle::framework::DataLayoutToString(
                                 LayoutAutoTune::Instance().GetDesiredLayout());
      } else {
        LayoutAutoTune::Instance().DisableLayoutAutoTune();
        return ins;
      }
    }
  }

  std::shared_ptr<LayoutTransposer<VarType>> transposer = nullptr;
  if (op_type == "conv2d") {
    transposer =
        std::make_shared<HeavilyLayoutSensitiveOpTransposer<VarType>>(op_type);
    transposer->SetArguments({"Input"}, {"Output"}, {"data_format"});
  } else if (op_type == "batch_norm") {
    transposer =
        std::make_shared<HeavilyLayoutSensitiveOpTransposer<VarType>>(op_type);
    transposer->SetArguments({"X"}, {"Y"}, {"data_layout"});
  } else if (op_type == "pool2d") {
    transposer =
        std::make_shared<HeavilyLayoutSensitiveOpTransposer<VarType>>(op_type);
    transposer->SetArguments({"X"}, {"Out"}, {"data_format"});
  } else if (op_type == "transpose2") {
    transposer = std::make_shared<TransposeOpTransposer<VarType>>(op_type);
  } else if (op_type == "flatten_contiguous_range") {
    transposer = std::make_shared<FlattenOpTransposer<VarType>>(op_type);
  } else if (op_type.find("elementwise_") != std::string::npos) {
    transposer = std::make_shared<ElementwiseOpTransposer<VarType>>(op_type);
  } else if (LayoutAutoTune::Instance().IsLayoutAgnostic(op_type)) {
    transposer = std::make_shared<LayoutTransposer<VarType>>(op_type);
  } else if (LayoutAutoTune::Instance().IsLightlyLayoutSensitive(op_type)) {
    transposer =
        std::make_shared<LightlyLayoutSensitiveOpTransposer<VarType>>(op_type);
  } else {
    PADDLE_ENFORCE_NOT_NULL(
        transposer, phi::errors::Unimplemented(
                        "%s 's LayoutTransposer is unimplemented.", op_type));
  }

  return transposer->Run(ins, outs, attrs, tracer);
}

}  // namespace autotune
}  // namespace phi
