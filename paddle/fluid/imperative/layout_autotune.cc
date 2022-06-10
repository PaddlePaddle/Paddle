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

#include "paddle/fluid/imperative/layout_autotune.h"
#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/imperative/layout_transformer.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/errors.h"

namespace paddle {
namespace imperative {

bool LayoutAutoTune::UseLayoutAutoTune() const {
#if defined(PADDLE_WITH_CUDA)
  if (!phi::backends::gpu::TensorCoreAvailable()) {
    LOG(INFO) << "Layout AutoTuning is not available.";
    return false;
  } else {
    return use_layout_autotune_;
  }
#else
  return false;
#endif
}

LayoutAutoTune::LayoutAutoTune() {
  const auto& op_info = paddle::framework::OpInfoMap::Instance().map();
  for (auto it = op_info.begin(); it != op_info.end(); it++) {
    // only record forwrd operators
    if (it->first.find("_grad") != std::string::npos) {
      continue;
    }

    // some normalization operators such as instance_norm and layer_norm
    // do not have data_format attr, but are layout sensitive.
    if (it->first.find("norm") != std::string::npos) {
      layout_agnostic_ops_.emplace(it->first);
      continue;
    }

    auto* attr_checker = it->second.Checker();
    if (attr_checker) {
      auto attrs = attr_checker->GetDefaultAttrMap();
      if (attrs.find("data_format") != attrs.end() ||
          attrs.find("data_layout") != attrs.end()) {
        VLOG(4) << "Heavily layout sensitive OP: " << it->first;
        heavily_layout_sensitive_ops_.emplace(it->first);
        continue;
      }

      // Attribute name is fuzzy matched, such as start and start_axis.
      bool layout_agnostic = true;
      for (auto& attr : attrs) {
        auto attr_name = attr.first;
        VLOG(6) << "OP: " << it->first << " Attr Name: " << attr_name;
        if (attr_name.find("axis") != std::string::npos ||
            attr_name.find("axes") != std::string::npos ||
            attr_name.find("dim") != std::string::npos ||
            attr_name.find("start") != std::string::npos ||
            attr_name.find("end") != std::string::npos) {
          VLOG(4) << "Lightly layout sensitive OP: " << it->first;
          layout_agnostic = false;
          lightly_layout_sensitive_ops_.emplace(it->first);
          break;
        }
      }

      if (layout_agnostic) {
        VLOG(4) << "Layout agnostic_ops: " << it->first;
        layout_agnostic_ops_.emplace(it->first);
      }
    }
  }

  VLOG(3) << "The number of layout agnostic OPs: "
          << layout_agnostic_ops_.size() << ", heavily layout sensitive OPs: "
          << heavily_layout_sensitive_ops_.size()
          << ", lightly layout sensitive OPs: "
          << lightly_layout_sensitive_ops_.size();
}

template <typename VarType>
paddle::imperative::NameVarMap<VarType> AutoTuneLayout(
    const std::string& op_type,
    const paddle::imperative::NameVarMap<VarType>& ins,
    const paddle::imperative::NameVarMap<VarType>& outs,
    paddle::framework::AttributeMap* attrs,
    const std::shared_ptr<imperative::Tracer>& tracer) {
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

  std::shared_ptr<LayoutTransformer<VarType>> transposer = nullptr;
  if (op_type == "conv2d") {
    transposer =
        std::make_shared<HeavilyLayoutSensitiveOpTransformer<VarType>>(op_type);
    transposer->SetArguments({"Input"}, {"Output"}, {"data_format"});
  } else if (op_type == "batch_norm") {
    transposer =
        std::make_shared<HeavilyLayoutSensitiveOpTransformer<VarType>>(op_type);
    transposer->SetArguments({"X"}, {"Y"}, {"data_layout"});
  } else if (op_type == "pool2d") {
    transposer =
        std::make_shared<HeavilyLayoutSensitiveOpTransformer<VarType>>(op_type);
    transposer->SetArguments({"X"}, {"Out"}, {"data_format"});
  } else if (op_type == "transpose2") {
    transposer = std::make_shared<TransposeOpTransformer<VarType>>(op_type);
  } else if (op_type == "flatten_contiguous_range") {
    transposer = std::make_shared<FlattenOpTransformer<VarType>>(op_type);
  } else if (op_type.find("elementwise_") != std::string::npos) {
    transposer = std::make_shared<ElementwiseOpTransformer<VarType>>(op_type);
  } else if (LayoutAutoTune::Instance().IsLayoutAgnostic(op_type)) {
    transposer = std::make_shared<LayoutTransformer<VarType>>(op_type);
  } else if (LayoutAutoTune::Instance().IsLightlyLayoutSensitive(op_type)) {
    transposer =
        std::make_shared<LightlyLayoutSensitiveOpTransformer<VarType>>(op_type);
  } else {
    PADDLE_ENFORCE_NOT_NULL(
        transposer, phi::errors::Unimplemented(
                        "%s 's LayoutTransformer is unimplemented.", op_type));
  }

  return transposer->Apply(ins, outs, attrs, tracer);
}
template paddle::imperative::NameVarMap<VarBase> AutoTuneLayout<VarBase>(
    const std::string& op_type,
    const paddle::imperative::NameVarMap<VarBase>& ins,
    const paddle::imperative::NameVarMap<VarBase>& outs,
    paddle::framework::AttributeMap* attrs,
    const std::shared_ptr<imperative::Tracer>& tracer);
template paddle::imperative::NameVarMap<egr::EagerVariable>
AutoTuneLayout<egr::EagerVariable>(
    const std::string& op_type,
    const paddle::imperative::NameVarMap<egr::EagerVariable>& ins,
    const paddle::imperative::NameVarMap<egr::EagerVariable>& outs,
    paddle::framework::AttributeMap* attrs,
    const std::shared_ptr<imperative::Tracer>& tracer);

}  // namespace imperative
}  // namespace paddle
