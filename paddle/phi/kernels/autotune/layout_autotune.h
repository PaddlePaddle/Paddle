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
#include "paddle/fluid/imperative/var_helper.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/errors.h"

namespace phi {
namespace autotune {

using DataLayout = paddle::experimental::DataLayout;

class LayoutAutoTune {
 public:
  static LayoutAutoTune& Instance() {
    static LayoutAutoTune layout_autoTune;
    return layout_autoTune;
  }

  bool UseLayoutAutoTune() { return use_layout_autotune_; }

  void EnableLayoutAutoTune() { use_layout_autotune_ = true; }

  void DisableLayoutAutoTune() { use_layout_autotune_ = false; }

  void Update() { layout_optimized_vars_.clear(); }

  void InsertOptimizedVars(const std::string name) {
    layout_optimized_vars_.insert(name);
    VLOG(4) << "insert var " << name;
  }

  bool NeedTuneLayout(const std::string name) {
    return layout_optimized_vars_.count(name) == 0;
  }

  bool IsHeavilyLayoutSensitive(const std::string& op_type) {
    return heavily_layout_sensitive_ops_.count(op_type) != 0;
  }

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
  LayoutAutoTune() = default;

  bool use_layout_autotune_ = false;

  std::unordered_set<std::string> layout_agnostic_ops_ = {"relu",
                                                          "elementwise_add"};

  // Both functionality and performance are affected by data layout.
  // Such as operators with data_format attribute.
  std::unordered_set<std::string> heavily_layout_sensitive_ops_ = {
      "conv2d", "batch_norm", "pool2d",
  };

  // The functionality may be affected layout transformation before them.
  // Such as operators with axis attribute.
  std::unordered_set<std::string> lightly_layout_sensitive_ops_ = {"mean"};

  std::unordered_set<std::string> layout_optimized_vars_;

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
    AddOptimizedVars(outs);
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
  void AddOptimizedVars(
      const paddle::imperative::NameVarMap<VarType>& outs) const {
    if (outs_.empty()) {
      for (auto& pair : outs) {
        for (auto& var : pair.second) {
          LayoutAutoTune::Instance().InsertOptimizedVars(
              paddle::imperative::GetNameFromVar(var));
        }
      }
    } else {
      for (auto& name : outs_) {
        auto out_vars = outs.at(name);
        for (auto& var : out_vars) {
          LayoutAutoTune::Instance().InsertOptimizedVars(
              paddle::imperative::GetNameFromVar(var));
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
    std::string dst_format = paddle::framework::DataLayoutToString(
        LayoutAutoTune::Instance().GetDesiredLayout());
    if (attrs->find("data_format") != attrs->end() &&
        boost::get<std::string>((*attrs)["data_format"]) != dst_format) {
      VLOG(4) << "Origin layout: "
              << boost::get<std::string>((*attrs)["data_format"])
              << ", Desired layout: " << dst_format;
      (*attrs)["data_format"] = std::string("NHWC");
    } else if (attrs->find("data_layout") != attrs->end() &&
               boost::get<std::string>((*attrs)["data_layout"]) != dst_format) {
      VLOG(4) << "Origin layout: "
              << boost::get<std::string>((*attrs)["data_layout"])
              << ", Desired layout: " << dst_format;
      (*attrs)["data_layout"] = std::string("NHWC");
    }

    // Step 2: Transpose op's input and record the transposed var
    for (auto& name : this->Inputs()) {
      auto& in_vars = new_ins[name];
      for (auto& var : in_vars) {
        auto origin_var = paddle::imperative::GetNameFromVar(var);
        if (LayoutAutoTune::Instance().NeedTuneLayout(origin_var)) {
          var = TraceTransposeOp(var, tracer);
          auto transposed_var = paddle::imperative::GetNameFromVar(var);
          LayoutAutoTune::Instance().InsertOptimizedVars(transposed_var);
          VLOG(4) << "Transpose " << origin_var << " to " << transposed_var;
        }
      }
    }

    // Step 3: record the Op's layout sensitive outs var
    this->AddOptimizedVars(outs);

    return new_ins;
  }

  std::shared_ptr<VarType> TraceTransposeOp(
      const std::shared_ptr<VarType>& var,
      const std::shared_ptr<paddle::imperative::Tracer>& tracer) {
    paddle::imperative::NameVarMap<VarType> ins = {{"X", {var}}};
    auto out =
        std::shared_ptr<VarType>(new VarType(tracer->GenerateUniqueName()));
    paddle::imperative::NameVarMap<VarType> outs = {{"Out", {out}}};
    std::vector<int> axis = {0, 2, 3, 1};
    paddle::framework::AttributeMap attrs = {{"axis", axis}};
    tracer->TraceOp("transpose2", ins, outs, std::move(attrs));
    return out;
  }
};

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
    VLOG(3) << "Optimze heavily layout sensitive op " << this->Type();
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
  } else {
    transposer = std::make_shared<LayoutTransposer<VarType>>(op_type);
    /*
    PADDLE_ENFORCE_NOT_NULL(transposer,
        phi::errors::Unimplemented("%s 's LayoutTransposer is unimplemented.",
    op_type));
    */
  }
  return transposer->Run(ins, outs, attrs, tracer);
}

}  // namespace autotune
}  // namespace phi
