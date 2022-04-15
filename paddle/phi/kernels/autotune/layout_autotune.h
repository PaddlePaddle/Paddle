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
#include <set>
#include "paddle/fluid/framework/var_type_traits.h"
#include "paddle/fluid/imperative/var_helper.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/errors.h"
#include "paddle/phi/kernels/autotune/layout_agnostic_ops.h"

namespace phi {
namespace autotune {

class DenseTensor;

using DataLayout = paddle::experimental::DataLayout;

template <typename VarType>
std::shared_ptr<VarType> TraceTransposeOp(
    const std::shared_ptr<VarType>& var,
    const std::vector<int>& axis,
    const std::shared_ptr<paddle::imperative::Tracer>& tracer) {
  paddle::imperative::NameVarMap<VarType> ins = {{"X", {var}}};
  auto out =
      std::shared_ptr<VarType>(new VarType(tracer->GenerateUniqueName()));
  paddle::imperative::NameVarMap<VarType> outs = {{"Out", {out}}};
  paddle::framework::AttributeMap attrs = {{"axis", axis}};
  tracer->TraceOp("transpose2", ins, outs, std::move(attrs));
  return out;
}

class LayoutAutoTune {
 public:
  static LayoutAutoTune& Instance() {
    static LayoutAutoTune layout_autoTune;
    return layout_autoTune;
  }

  bool UseLayoutAutoTune() { return use_layout_autotune_; }

  void EnableLayoutAutoTune() { use_layout_autotune_ = true; }

  void DisableLayoutAutoTune() { use_layout_autotune_ = false; }

  bool IsLightlyLayoutSensitive(const std::string& op_type) {
    return lightly_layout_sensitive_ops_.count(op_type) != 0;
  }

  bool IsLayoutAgnostic(const std::string& op_type) {
    return layout_agnostic_ops_.count(op_type) != 0;
  }

  void SetAgnosticOps(const std::unordered_set<std::string>& agnostic_ops) {
    layout_agnostic_ops_ = agnostic_ops;
  }

  DataLayout GetDesiredLayout() { return layout_; }

  void SetDesiredLayout(const DataLayout& layout) { layout_ = layout; }

 private:
  LayoutAutoTune() {
    layout_agnostic_ops_ =
        paddle::imperative::LayoutAutotuneOperators::Instance()
            .GetAgnosticOps();
  }

  bool use_layout_autotune_ = false;

  std::unordered_set<std::string> layout_agnostic_ops_;

  std::unordered_set<std::string> lightly_layout_sensitive_ops_ = {
      "reduce_max", "reduce_min", "reduce_mean", "softmax_with_cross_entropy"};

  DataLayout layout_ = DataLayout::UNDEFINED;
};

template <typename VarType>
class LayoutTransposer {
 public:
  explicit LayoutTransposer(const std::string& type) : type_(type) {}

  virtual ~LayoutTransposer() {}

  LayoutTransposer(const LayoutTransposer&) = delete;
  LayoutTransposer& operator=(const LayoutTransposer&) = delete;

  virtual paddle::imperative::NameVarMap<VarType> Run(
      const paddle::imperative::NameVarMap<VarType>& ins,
      const paddle::imperative::NameVarMap<VarType>& outs,
      paddle::framework::AttributeMap* attrs,
      const std::shared_ptr<paddle::imperative::Tracer>& tracer) {
    VLOG(3) << "Optimze Layout agnostic op: " << type_;
    auto in_layout = DataLayout::UNDEFINED;
    for (auto& pair : ins) {
      for (auto& var : pair.second) {
        // once the any input is desired layout, we set in_layout is desired
        // layout.
        if (paddle::imperative::GetDataLayout(var) ==
            LayoutAutoTune::Instance().GetDesiredLayout()) {
          in_layout = LayoutAutoTune::Instance().GetDesiredLayout();
          break;
        }
      }
    }
    SetVarsLayout(outs, in_layout);
    return ins;
  }

  // Set inputs, outputs and attributes to be optimized for the transposer.
  // Those may respectively be a subset of the corresponding original argument
  // of the operator.
  void SetArguments(const std::vector<std::string>& ins,
                    const std::vector<std::string>& outs,
                    const std::vector<std::string>& attrs) {
    ins_ = ins;
    outs_ = outs;
    attrs_ = attrs;
  }

  // Record the variables that have been optimized so that they will
  // not be repeatedly transformed later.
  // If outs_ is not specified, it means all outputs of the operator
  // will be recorded. Otherwise, it only records the specified outputs.
  void SetVarsLayout(const paddle::imperative::NameVarMap<VarType>& outs,
                     DataLayout layout) const {
    if (outs_.empty()) {
      for (auto& pair : outs) {
        for (auto& var : pair.second) {
          SetDataLayout(var, layout);
        }
      }
    } else {
      for (auto& name : outs_) {
        auto out_vars = outs.at(name);
        for (auto& var : out_vars) {
          SetDataLayout(var, layout);
        }
      }
    }
  }

  const std::vector<std::string>& Inputs() const { return ins_; }
  const std::vector<std::string>& Outputs() const { return outs_; }
  const std::vector<std::string>& Attributes() const { return attrs_; }

  const std::string& Type() { return type_; }

 protected:
  std::string type_;
  std::vector<std::string> ins_;
  std::vector<std::string> outs_;
  std::vector<std::string> attrs_;
};

/*
 * Both functionality and performance are affected by data layout.
 * Such as operators with data_format attribute.
 */
template <typename VarType>
class HeavilyLayoutSensitiveOpTransposer : public LayoutTransposer<VarType> {
 public:
  explicit HeavilyLayoutSensitiveOpTransposer(const std::string& type)
      : LayoutTransposer<VarType>(type) {}

  paddle::imperative::NameVarMap<VarType> Run(
      const paddle::imperative::NameVarMap<VarType>& ins,
      const paddle::imperative::NameVarMap<VarType>& outs,
      paddle::framework::AttributeMap* attrs,
      const std::shared_ptr<paddle::imperative::Tracer>& tracer) {
    VLOG(3) << "Optimze heavily layout sensitive op " << this->Type();
    paddle::imperative::NameVarMap<VarType> new_ins(ins);

    // Step 1: Adjust the data_layout attr to the desired layout
    auto desired_layout = LayoutAutoTune::Instance().GetDesiredLayout();
    std::string desired_layout_str = paddle::framework::DataLayoutToString(
        LayoutAutoTune::Instance().GetDesiredLayout());
    if (attrs->find("data_format") != attrs->end() &&
        boost::get<std::string>((*attrs)["data_format"]) !=
            desired_layout_str) {
      VLOG(4) << "Origin layout: "
              << boost::get<std::string>((*attrs)["data_format"])
              << ", Desired layout: " << desired_layout_str;
      (*attrs)["data_format"] = desired_layout_str;
    } else if (attrs->find("data_layout") != attrs->end() &&
               boost::get<std::string>((*attrs)["data_layout"]) !=
                   desired_layout_str) {
      VLOG(4) << "Origin layout: "
              << boost::get<std::string>((*attrs)["data_layout"])
              << ", Desired layout: " << desired_layout_str;
      (*attrs)["data_layout"] = desired_layout_str;
    }

    // Step 2: Transpose op's input and record the transposed var
    for (auto& name : this->Inputs()) {
      auto& in_vars = new_ins[name];
      for (auto& var : in_vars) {
        auto var_layout = paddle::imperative::GetDataLayout(var);
        VLOG(4) << "Origin Var: " << paddle::imperative::GetNameFromVar(var)
                << ", Layout: "
                << paddle::framework::DataLayoutToString(var_layout);
        if (var_layout != desired_layout) {
          // NCHW -> NHWC
          std::vector<int> axis = {0, 2, 3, 1};
          var = TraceTransposeOp(var, axis, tracer);
          SetDataLayout(var, desired_layout);
          VLOG(4) << "Transposed Var: "
                  << paddle::imperative::GetNameFromVar(var)
                  << ", Layout: " << desired_layout_str;
        }
      }
    }

    // Step 3: record the Op's layout sensitive outs var
    this->SetVarsLayout(outs, desired_layout);

    return new_ins;
  }
};

/*
 * The functionality may be affected layout transformation before them.
 * Such as operators with axis attribute.
 */
template <typename VarType>
class LightlyLayoutSensitiveOpTransposer : public LayoutTransposer<VarType> {
 public:
  explicit LightlyLayoutSensitiveOpTransposer(const std::string& type)
      : LayoutTransposer<VarType>(type) {}

  paddle::imperative::NameVarMap<VarType> Run(
      const paddle::imperative::NameVarMap<VarType>& ins,
      const paddle::imperative::NameVarMap<VarType>& outs,
      paddle::framework::AttributeMap* attrs,
      const std::shared_ptr<paddle::imperative::Tracer>& tracer) {
    VLOG(3) << "Optimze lightly layout sensitive op " << this->Type();
    paddle::imperative::NameVarMap<VarType> new_ins(ins);
    // If input's layout is not tuned, transformation is unnecessary.
    // If input's layout is already tuned, it will be transformed back to NCHW.
    // TODO(zhiqiang): The op of this type should be adapted to the previous
    // operator output data layout. Currently only a few operators are
    // supported,
    // and transposers need to be carefully designed to ensure that they do not
    // cause exceptions.
    for (auto& name : this->Inputs()) {
      auto& in_vars = new_ins[name];
      for (auto& var : in_vars) {
        auto var_layout = paddle::imperative::GetDataLayout(var);
        if (var_layout == LayoutAutoTune::Instance().GetDesiredLayout()) {
          VLOG(4) << "Transpose " << paddle::imperative::GetNameFromVar(var)
                  << "to NCHW";
          // NHWC -> NCHW
          std::vector<int> axis = {0, 3, 1, 2};
          var = TraceTransposeOp(var, axis, tracer);
          SetDataLayout(var, DataLayout::NCHW);
        }
      }
    }
    return new_ins;
  }
};

template <typename VarType>
class TransposeOpTransposer
    : public LightlyLayoutSensitiveOpTransposer<VarType> {
 public:
  explicit TransposeOpTransposer(const std::string& type)
      : LightlyLayoutSensitiveOpTransposer<VarType>(type) {}

  paddle::imperative::NameVarMap<VarType> Run(
      const paddle::imperative::NameVarMap<VarType>& ins,
      const paddle::imperative::NameVarMap<VarType>& outs,
      paddle::framework::AttributeMap* attrs,
      const std::shared_ptr<paddle::imperative::Tracer>& tracer) {
    VLOG(3) << "Optimze lightly layout sensitive op " << this->Type();
    // When the input layout has not been tuned and desired format is NHWC,
    // the output of the current transpose op needs to be marked as tuned.
    // Otherwise, it will fall back to the LightlyLayoutSensitiveOpTransposer,
    // Because this means that there is a transpose layer in the network, it
    // is better to transpose the result to the original format.
    auto axis = boost::get<std::vector<int>>((*attrs)["axis"]);
    // NCHW->NHWC
    std::vector<int> perm = {0, 2, 3, 1};
    auto& in_var = ins.at("X")[0];
    auto var_layout = paddle::imperative::GetDataLayout(in_var);
    if (var_layout != LayoutAutoTune::Instance().GetDesiredLayout() &&
        axis == perm) {
      this->SetVarsLayout(outs, LayoutAutoTune::Instance().GetDesiredLayout());
      return ins;
    } else {
      return LightlyLayoutSensitiveOpTransposer<VarType>::Run(
          ins, outs, attrs, tracer);
    }
  }
};

template <typename VarType>
class FlattenOpTransposer : public LightlyLayoutSensitiveOpTransposer<VarType> {
 public:
  explicit FlattenOpTransposer(const std::string& type)
      : LightlyLayoutSensitiveOpTransposer<VarType>(type) {}

  paddle::imperative::NameVarMap<VarType> Run(
      const paddle::imperative::NameVarMap<VarType>& ins,
      const paddle::imperative::NameVarMap<VarType>& outs,
      paddle::framework::AttributeMap* attrs,
      const std::shared_ptr<paddle::imperative::Tracer>& tracer) {
    VLOG(3) << "Optimze lightly layout sensitive op " << this->Type();
    // Flatten the C, H, W dimensions will not affect functionality.
    // Transformation is unnecessary. But in other cases, it needs to
    // fall back to the LightlyLayoutSensitiveOpTransposer.
    auto start_axis = boost::get<int>((*attrs)["start_axis"]);
    auto stop_axis = boost::get<int>((*attrs)["stop_axis"]);
    // TODO(zhangting): Rank checker need be added.
    if (start_axis == 1 && stop_axis == 3) {
      return ins;
      this->SetVarsLayout(outs, DataLayout::UNDEFINED);
    } else {
      return LightlyLayoutSensitiveOpTransposer<VarType>::Run(
          ins, outs, attrs, tracer);
    }
  }
};

template <typename VarType>
paddle::imperative::NameVarMap<VarType> LayoutOptimizer(
    const std::string& op_type,
    const paddle::imperative::NameVarMap<VarType>& ins,
    const paddle::imperative::NameVarMap<VarType>& outs,
    paddle::framework::AttributeMap* attrs,
    const paddle::platform::Place& place,
    const std::shared_ptr<paddle::imperative::Tracer>& tracer) {
  if (!LayoutAutoTune::Instance().UseLayoutAutoTune()) {
    return ins;
  }

  // When layout autotune is enabled, the tuner will check the desired layout.
  // (1) If the desired layout is undefined, and there is no convolutional
  // layers,
  // layout optimization is unnecessary. Otherwise, the desired layout will be
  // set
  // to the best layout only when these is a convolutional layer with
  // NCHW-Layout
  // and the TensorCore is available. The defined layout is undefined by
  // default.
  // (2) If the desired layout is defined, run the transposer.

  if (LayoutAutoTune::Instance().GetDesiredLayout() == DataLayout::UNDEFINED) {
    // Layout autotune only supports model with convolutional layers
    if (op_type != "conv2d") {
      return ins;
    } else {
      if (boost::get<std::string>((*attrs)["data_format"]) == "NCHW" &&
          paddle::platform::is_gpu_place(place) &&
          phi::backends::gpu::TensorCoreAvailable()) {
        LayoutAutoTune::Instance().SetDesiredLayout(DataLayout::NHWC);
        VLOG(3) << "Tune the layout from "
                << boost::get<std::string>((*attrs)["data_format"]) << " to "
                << paddle::framework::DataLayoutToString(
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
  } else if (LayoutAutoTune::Instance().IsLayoutAgnostic(op_type)) {
    transposer = std::make_shared<LayoutTransposer<VarType>>(op_type);
  } else if (LayoutAutoTune::Instance().IsLightlyLayoutSensitive(op_type)) {
    transposer =
        std::make_shared<LightlyLayoutSensitiveOpTransposer<VarType>>(op_type);
  } else {
    PADDLE_ENFORCE_NOT_NULL(
        transposer,
        phi::errors::Unimplemented("%s 's LayoutTransposer is unimplemented.",
                                   op_type));
  }

  return transposer->Run(ins, outs, attrs, tracer);
}

}  // namespace autotune
}  // namespace phi
