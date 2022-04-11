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

enum class TransMode {
  DEFAULT = 0,
  NCHWToNHWC,
  NHWCToNHWC,
};

// constexpr char kNCHW[] = "NCHW";
// constexpr char kNHWC[] = "NHWC";

class LayoutTransposer {
 public:
  explicit LayoutTransposer() {}
  virtual ~LayoutTransposer() {}

  LayoutTransposer(const LayoutTransposer&) = delete;
  LayoutTransposer& operator=(const LayoutTransposer&) = delete;

  void virtual Tranpose(paddle::framework::AttributeMap* attrs) {}
};

class LayoutSensitiveOpTransposer : public LayoutTransposer {
 public:
  explicit LayoutSensitiveOpTransposer() : LayoutTransposer() {}
  void Tranpose(paddle::framework::AttributeMap* attrs,
                const std::string& dst_format) {
    if (attrs->find("data_format") != attrs->end() &&
        boost::get<std::string>((*attrs)["data_format"]) != dst_format) {
      (*attrs)["data_format"] = std::string("NHWC");
    } else if (attrs->find("data_layout") != attrs->end() &&
               boost::get<std::string>((*attrs)["data_layout"]) != dst_format) {
      (*attrs)["data_layout"] = std::string("NHWC");
    }
  }
};

template <typename VarType>
class LayoutAutoTune {
 public:
  static LayoutAutoTune& Instance() {
    static LayoutAutoTune<VarType> layout_autoTune;
    return layout_autoTune;
  }

  TransMode LayoutTransMode() { return TransMode::NHWCToNHWC; }

  paddle::imperative::NameVarMap<VarType> OptimizeLayout(
      const std::string& type,
      const paddle::imperative::NameVarMap<VarType>& ins,
      const paddle::imperative::NameVarMap<VarType>& outs,
      paddle::framework::AttributeMap* attrs,
      const std::shared_ptr<paddle::imperative::Tracer>& tracer) {
    paddle::imperative::NameVarMap<VarType> new_ins(ins);
    if (heavily_layout_sensitive_ops_.count(type) != 0) {
      LayoutSensitiveOpTransposer transposer;
      std::string dst_format = "NHWC";
      transposer.Tranpose(attrs, dst_format);
      VLOG(3) << "Change the origin data_fommat to " << dst_format;
      std::string in_name = "X";
      if (new_ins.find("Input") != new_ins.end()) {
        in_name = "Input";
      }
      auto& var = new_ins[in_name][0];
      auto var_name = paddle::imperative::GetNameFromVar(var);
      VLOG(3) << "Op input name: " << var_name;
      if (NeedOptimized(var_name)) {
        var = TraceTransposeOp(var, tracer);
        VLOG(3) << "Transpose " << var_name << " to " << dst_format;
      }
      VLOG(3) << "Optimze heavily layout sensitive op " << type;
    } else if (lightly_layout_sensitive_ops_.count(type) != 0) {
      VLOG(3) << "Optimze lightly layout sensitive op's layout from " << type;
    } else if (layout_agnostic_ops_.count(type) != 0) {
      VLOG(3) << "Layout agnostic op: " << type;
    } else {
      VLOG(3) << "Unsupported op: " << type;
    }

    // add Op's outs var in set
    for (auto& pair : outs) {
      for (auto& var : pair.second) {
        InsertOptimizedVars(paddle::imperative::GetNameFromVar(var));
      }
    }
    return new_ins;
  }

  void InsertOptimizedVars(const std::string name) {
    layout_optimized_vars.insert(name);
    VLOG(3) << "insert var " << name;
  }

  bool NeedOptimized(const std::string name) {
    return layout_optimized_vars.count(name) == 0;
  }

  std::shared_ptr<VarType> TraceTransposeOp(
      const std::shared_ptr<VarType>& var,
      const std::shared_ptr<paddle::imperative::Tracer>& tracer) {
    paddle::imperative::NameVarMap<VarType> ins = {{"X", {var}}};
    auto var_name = tracer->GenerateUniqueName();
    auto out = std::shared_ptr<VarType>(new VarType(var_name));
    InsertOptimizedVars(var_name);
    paddle::imperative::NameVarMap<VarType> outs = {{"Out", {out}}};
    std::vector<int> axis = {0, 2, 3, 1};
    paddle::framework::AttributeMap attrs = {{"axis", axis}};
    tracer->TraceOp("transpose2", ins, outs, std::move(attrs));
    return out;
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

  std::unordered_set<std::string> layout_optimized_vars;
};

}  // namespace autotune
}  // namespace phi
