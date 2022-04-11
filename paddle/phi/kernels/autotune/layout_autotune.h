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
// #include "paddle/phi/core/compat/type_defs.h"
// #include "paddle/fluid/imperative/tracer.h"

namespace phi {
namespace autotune {

class LayoutAutoTune;

constexpr char kNCHW[] = "NCHW";
constexpr char kNHWC[] = "NHWC";

enum class TransMode {
  DEFAULT = 0,
  NCHWToNHWC,
  NHWCToNHWC,
};

class LayoutAutoTune {
 public:
  static LayoutAutoTune& Instance() {
    static LayoutAutoTune layout_autoTune;
    return layout_autoTune;
  }

  TransMode LayoutTransMode() { return TransMode::NHWCToNHWC; }

  void InsertOptimizedVars(const std::string name) {
    layout_optimized_vars_.insert(name);
    VLOG(3) << "insert var " << name;
  }

  bool NeedOptimized(const std::string name) {
    return layout_optimized_vars_.count(name) == 0;
  }

 private:
  LayoutAutoTune() = default;

  std::unordered_set<std::string> layout_agnostic_ops_ = {"relu",
                                                          "elementwise_add"};

  // Both functionality and performance are affected by data layout.
  // Such as operators with data_format attribute.
  std::unordered_set<std::string> heavily_layout_sensitive_ops_ = {
      "conv2d", "batch_norm",
      // "pool2d",
  };

  // The functionality may be affected layout transformation before them.
  // Such as operators with axis attribute.
  std::unordered_set<std::string> lightly_layout_sensitive_ops_ = {"mean"};

  std::unordered_set<std::string> layout_optimized_vars_;
};

class LayoutTransposer {
 public:
  explicit LayoutTransposer(const std::string& type)
      : type_(type) {}

  LayoutTransposer(const LayoutTransposer&) = delete;
  LayoutTransposer& operator=(const LayoutTransposer&) = delete;

  template <typename VarType>
  paddle::imperative::NameVarMap<VarType> Run(
      const paddle::imperative::NameVarMap<VarType>& ins,
      const paddle::imperative::NameVarMap<VarType>& outs,
      paddle::framework::AttributeMap* attrs,
      const std::shared_ptr<paddle::imperative::Tracer>& tracer) {
    VLOG(3) << "Optimze Layout agnostic op: " << type_;
    return ins;
  }

 protected:
  std::string type_;
};

class LayoutSensitiveOpTransposer : public LayoutTransposer {
 public:
  explicit LayoutSensitiveOpTransposer(
       const std::string& type) : LayoutTransposer(type) {}
  
  template <typename VarType>
  paddle::imperative::NameVarMap<VarType> Run(
      const paddle::imperative::NameVarMap<VarType>& ins,
      const paddle::imperative::NameVarMap<VarType>& outs,
      paddle::framework::AttributeMap* attrs,
      const std::shared_ptr<paddle::imperative::Tracer>& tracer) {
    VLOG(3) << "Optimze heavily layout sensitive op " << type_;
    paddle::imperative::NameVarMap<VarType> new_ins(ins);
    
    // step 1: reset the data_layout attr to the autotuning layout
    std::string dst_format = "NHWC";
    if (attrs->find("data_format") != attrs->end() &&
        boost::get<std::string>((*attrs)["data_format"]) != dst_format) {
      (*attrs)["data_format"] = std::string("NHWC");
    } else if (attrs->find("data_layout") != attrs->end() &&
               boost::get<std::string>((*attrs)["data_layout"]) != dst_format) {
      (*attrs)["data_layout"] = std::string("NHWC");
    }
    
    VLOG(3) << "Change the origin data_fommat to " << dst_format;

    // step 2: Transpose op's input
    for (auto &name : ins_) {
      auto in_vars = new_ins[name];
      for (auto &var : in_vars) {
        auto var_name = paddle::imperative::GetNameFromVar(var);
        VLOG(3) << "Op input name: " << var_name;  
        if (LayoutAutoTune::Instance().NeedOptimized(var_name)) {
          var = TraceTransposeOp(var, tracer);
          VLOG(3) << "Transpose " << var_name << " to " << dst_format;
        }
      }
    }

    // add Op's outs var in set
    for (auto& pair : outs) {
      for (auto& var : pair.second) {
        LayoutAutoTune::Instance().InsertOptimizedVars(paddle::imperative::GetNameFromVar(var));
      }
    }
    return new_ins;
  }

  template <typename VarType>
  std::shared_ptr<VarType> TraceTransposeOp(
      const std::shared_ptr<VarType>& var,
      const std::shared_ptr<paddle::imperative::Tracer>& tracer) {
    paddle::imperative::NameVarMap<VarType> ins = {{"X", {var}}};
    auto var_name = tracer->GenerateUniqueName();
    auto out = std::shared_ptr<VarType>(new VarType(var_name));
    LayoutAutoTune::Instance().InsertOptimizedVars(var_name);
    paddle::imperative::NameVarMap<VarType> outs = {{"Out", {out}}};
    std::vector<int> axis = {0, 2, 3, 1};
    paddle::framework::AttributeMap attrs = {{"axis", axis}};
    tracer->TraceOp("transpose2", ins, outs, std::move(attrs));
    return out;
  }

  void SetInputs(const std::vector<std::string>& ins) {
    ins_ = ins;
  }

  void SetOutputs(const std::vector<std::string>& outs) {
    outs_ = outs;
  }

  void SetAttributes(const std::vector<std::string>& attrs) {
    attrs_ = attrs;
  }

  private:
   std::string type_;
   std::vector<std::string> ins_;
   std::vector<std::string> outs_;
   std::vector<std::string> attrs_;
};

class ConvLayoutTransposer : public LayoutSensitiveOpTransposer {
 public:
  explicit ConvLayoutTransposer(const std::string& type) : LayoutSensitiveOpTransposer(type) {
    SetInputs({"Input"});
    SetOutputs({"Output"});
    SetAttributes({"data_format"});
  }
};

class BatchNormLayoutTransposer : public LayoutSensitiveOpTransposer {
 public:
  explicit BatchNormLayoutTransposer(const std::string& type) : LayoutSensitiveOpTransposer(type) {
    SetInputs({"X"});
    SetOutputs({"Y"});
    SetAttributes({"data_layout"});
  }
};

std::shared_ptr<LayoutTransposer> GetLayoutTransposer(const std::string& op_type) {
  std::shared_ptr<LayoutTransposer> transposer = nullptr;
  VLOG(3) << "GetLayoutTransposer for " << op_type;
  if (op_type == "conv2d") {
    transposer = std::make_shared<ConvLayoutTransposer>(op_type);
  } else if (op_type == "batch_norm") {
    transposer = std::make_shared<BatchNormLayoutTransposer>(op_type);
  } else {
    transposer = std::make_shared<LayoutTransposer>(op_type);
  }

  return transposer;
}

}  // namespace autotune
}  // namespace phi
