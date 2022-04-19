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
#include "paddle/fluid/imperative/tracer.h"
#include "paddle/fluid/imperative/var_helper.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/errors.h"

namespace phi {
namespace autotune {

template <typename VarType>
std::shared_ptr<VarType> TraceTransposeOp(
    const std::shared_ptr<VarType>& var, const DataLayout layout,
    const std::shared_ptr<paddle::imperative::Tracer>& tracer) {
  std::vector<int> axis;
  if (layout == DataLayout::NHWC) {
    axis = {0, 2, 3, 1};
  } else if (layout == DataLayout::NCHW) {
    axis = {0, 3, 1, 2};
  } else {
    axis = {0, 1, 2, 3};
  }
  paddle::imperative::NameVarMap<VarType> ins = {{"X", {var}}};
  auto out =
      std::shared_ptr<VarType>(new VarType(tracer->GenerateUniqueName()));
  auto x_shape =
      std::shared_ptr<VarType>(new VarType(tracer->GenerateUniqueName()));
  paddle::imperative::NameVarMap<VarType> outs = {{"Out", {out}},
                                                  {"XShape", {x_shape}}};
  paddle::framework::AttributeMap attrs = {{"axis", axis}};
  tracer->TraceOp("transpose2", ins, outs, std::move(attrs));
  paddle::imperative::SetDataLayout(out, layout);
  VLOG(4) << "Transpose " << paddle::imperative::GetNameFromVar(var) << "["
          << paddle::framework::DataLayoutToString(
                 paddle::imperative::GetDataLayout(var))
          << "]"
          << " to " << paddle::imperative::GetNameFromVar(out) << "["
          << paddle::framework::DataLayoutToString(
                 paddle::imperative::GetDataLayout(out))
          << "]";
  return out;
}

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
        // Once the any input is desired layout, we set in_layout is desired
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

  // Set the variables's layout to the specified layout.
  // If outs_ is not specified, it means all outputs of the operator
  // will be considered. Otherwise, it only set layout for the specified output.
  void SetVarsLayout(const paddle::imperative::NameVarMap<VarType>& outs,
                     DataLayout layout) const {
    if (outs_.empty()) {
      for (auto& pair : outs) {
        for (auto& var : pair.second) {
          paddle::imperative::SetDataLayout(var, layout);
        }
      }
    } else {
      for (auto& name : outs_) {
        auto out_vars = outs.at(name);
        for (auto& var : out_vars) {
          paddle::imperative::SetDataLayout(var, layout);
        }
      }
    }
  }

  const std::vector<std::string>& Inputs() const { return ins_; }
  const std::vector<std::string>& Outputs() const { return outs_; }
  const std::vector<std::string>& Attributes() const { return attrs_; }

  const std::string& Type() { return type_; }

 protected:
  std::string type_{};
  std::vector<std::string> ins_{};
  std::vector<std::string> outs_{};
  std::vector<std::string> attrs_{};
};

template <typename VarType>
class ElementwiseOpTransposer : public LayoutTransposer<VarType> {
 public:
  explicit ElementwiseOpTransposer(const std::string& type)
      : LayoutTransposer<VarType>(type) {}

  paddle::imperative::NameVarMap<VarType> Run(
      const paddle::imperative::NameVarMap<VarType>& ins,
      const paddle::imperative::NameVarMap<VarType>& outs,
      paddle::framework::AttributeMap* attrs,
      const std::shared_ptr<paddle::imperative::Tracer>& tracer) {
    // [Why we need the this?]
    // The Elementwise Ops has a axis attr, it is to support broadcast.
    // When bias_attr of Conv is not false, the elementwise_add will be
    // appended, and the axis will be set to the channel dimension.

    // If the axis is set to the channel dimension, the attr transformation
    // is necessary. Otherwise, it will fall back to the LayoutTransposer::Run.
    auto desired_layout = LayoutAutoTune::Instance().GetDesiredLayout();
    if (attrs->find("axis") != attrs->end() &&
        BOOST_GET_CONST(int, (*attrs)["axis"]) != -1) {
      VLOG(3) << "Optimze layout agnostic op " << this->Type();
      if (desired_layout == DataLayout::NHWC) {
        (*attrs)["axis"] = 3;
      } else if (desired_layout == DataLayout::NCHW) {
        (*attrs)["axis"] = 1;
      } else {
        PADDLE_ENFORCE_EQ(
            desired_layout, DataLayout::UNDEFINED,
            phi::errors::PreconditionNotMet("DataLayout is unsupport."));
      }
      this->SetVarsLayout(outs, desired_layout);
      return ins;
    } else {
      return LayoutTransposer<VarType>::Run(ins, outs, attrs, tracer);
    }
  }
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
        BOOST_GET_CONST(std::string, (*attrs)["data_format"]) !=
            desired_layout_str) {
      VLOG(4) << "Origin layout attr: "
              << BOOST_GET_CONST(std::string, (*attrs)["data_format"])
              << ", Desired layout attr: " << desired_layout_str;
      (*attrs)["data_format"] = desired_layout_str;
    } else if (attrs->find("data_layout") != attrs->end() &&
               BOOST_GET_CONST(std::string, (*attrs)["data_layout"]) !=
                   desired_layout_str) {
      VLOG(4) << "Origin layout attr: "
              << BOOST_GET_CONST(std::string, (*attrs)["data_layout"])
              << ", Desired layout attr: " << desired_layout_str;
      (*attrs)["data_layout"] = desired_layout_str;
    }

    // Step 2: Transpose the specified input for Op and set the transposed var's
    // layout.
    for (auto& name : this->Inputs()) {
      auto& in_vars = new_ins[name];
      for (auto& var : in_vars) {
        auto var_layout = paddle::imperative::GetDataLayout(var);
        if (var_layout != desired_layout) {
          var = TraceTransposeOp(var, DataLayout::NHWC, tracer);
        }
      }
    }

    // Step 3: Set the Op's layout sensitive outs var.
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
    // TODO(zhangting): The op of this type should be adapted to the previous
    // operator output data layout. Currently only a few operators are
    // supported, and transposers need to be carefully designed to ensure that
    // they do not cause exceptions.
    for (auto& pair : new_ins) {
      for (auto& var : pair.second) {
        auto var_layout = paddle::imperative::GetDataLayout(var);
        if (var_layout == LayoutAutoTune::Instance().GetDesiredLayout()) {
          // Set layout to UNDEFINED so that TransposeOpTransposer do
          // NHWC->NCHW transformation.
          var = TraceTransposeOp(var, DataLayout::UNDEFINED, tracer);
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
    // When the input layout is the desired format, it means that there
    // is a transpose layer in the network, it is better to transpose
    // the result to the original format.
    // Instead of actually inserting a transpose Op, we fuse the inserted
    // transpose Op with the current transpose Op by transforming 'axis' attr.
    auto& in_var = ins.at("X")[0];
    auto var_layout = paddle::imperative::GetDataLayout(in_var);
    if (var_layout == LayoutAutoTune::Instance().GetDesiredLayout()) {
      auto axis = BOOST_GET_CONST(std::vector<int>, (*attrs)["axis"]);
      // NHWC->NCHW, permutaion will be set as follows.
      std::vector<int> perm = {0, 3, 1, 2};
      // fuse the transpose Ops by transforming axis.
      std::vector<int> fusion_axis = {perm[axis[0]], perm[axis[1]],
                                      perm[axis[2]], perm[axis[3]]};
      (*attrs)["axis"] = fusion_axis;
    }
    return ins;
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
    // So transformation is unnecessary. But in other cases, it needs to
    // fall back to the LightlyLayoutSensitiveOpTransposer.
    auto start_axis = BOOST_GET_CONST(int, (*attrs)["start_axis"]);
    auto stop_axis = BOOST_GET_CONST(int, (*attrs)["stop_axis"]);
    if (paddle::imperative::GetDataLayout(ins.at("X")[0]) ==
            LayoutAutoTune::Instance().GetDesiredLayout() &&
        start_axis == 1 && stop_axis == 3) {
      return ins;
    } else {
      return LightlyLayoutSensitiveOpTransposer<VarType>::Run(ins, outs, attrs,
                                                              tracer);
    }
  }
};

}  // namespace autotune
}  // namespace phi
