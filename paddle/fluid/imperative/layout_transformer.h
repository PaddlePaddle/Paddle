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
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/imperative/layout_autotune.h"
#include "paddle/fluid/imperative/tracer.h"
#include "paddle/fluid/imperative/var_helper.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/errors.h"
#include "paddle/phi/core/tensor_utils.h"
namespace paddle {
namespace imperative {
template <typename VarType>
void SetOutDataLayout(std::shared_ptr<VarType> var,
                      const paddle::experimental::DataLayout layout) {
  if (var != nullptr) {
    paddle::imperative::SetDataLayout(var, layout);
    // set out_tensor's layout
    if (var->MutableVar()->IsInitialized()) {
      paddle::framework::Variable* tmp_var = var->MutableVar();
      auto* out = tmp_var->GetMutable<framework::LoDTensor>();
      phi::DenseTensorUtils::GetMutableMeta(
          static_cast<framework::LoDTensor*>(out))
          ->layout = layout;
    }
  }
}

template <typename VarType>
std::shared_ptr<VarType> TraceTransposeOp(
    const std::shared_ptr<VarType>& var,
    const DataLayout layout,
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
class LayoutTransformer {
 public:
  explicit LayoutTransformer(const std::string& type) : type_(type) {}

  virtual ~LayoutTransformer() {}

  LayoutTransformer(const LayoutTransformer&) = delete;
  LayoutTransformer& operator=(const LayoutTransformer&) = delete;

  virtual paddle::imperative::NameVarMap<VarType> Apply(
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
        if (in_layout == DataLayout::UNDEFINED) {
          in_layout = paddle::imperative::GetDataLayout(var);
        }
        if (var != nullptr && (paddle::imperative::GetDataLayout(var) ==
                               LayoutAutoTune::Instance().GetDesiredLayout())) {
          in_layout = LayoutAutoTune::Instance().GetDesiredLayout();
          break;
        }
      }
    }
    VLOG(3) << "Optimze Layout agnostic op: " << type_ << " "
            << paddle::framework::DataLayoutToString(in_layout);
    if (in_layout != DataLayout::UNDEFINED) {
      SetVarsLayout(outs, in_layout);
    }
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
    bool not_in_out = true;
    if (!outs_.empty()) {
      for (auto& name : outs_) {
        if (outs.find(name) != outs.end()) {
          auto out_vars = outs.at(name);
          for (auto& var : out_vars) {
            if (var != nullptr) {
              paddle::imperative::SetOutDataLayout(var, layout);
            }
          }
          not_in_out = false;
        }
      }
    }

    if (not_in_out) {
      for (auto& pair : outs) {
        for (auto& var : pair.second) {
          if (var != nullptr) {
            paddle::imperative::SetOutDataLayout(var, layout);
          }
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

/*
 * Both functionality and performance are affected by data layout.
 * Such as operators with data_format attribute.
 */
template <typename VarType>
class HeavilyLayoutSensitiveOpTransformer : public LayoutTransformer<VarType> {
 public:
  explicit HeavilyLayoutSensitiveOpTransformer(const std::string& type)
      : LayoutTransformer<VarType>(type) {}

  paddle::imperative::NameVarMap<VarType> Apply(
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
        PADDLE_GET_CONST(std::string, (*attrs)["data_format"]) !=
            desired_layout_str) {
      VLOG(4) << "Origin layout attr: "
              << PADDLE_GET_CONST(std::string, (*attrs)["data_format"])
              << ", Desired layout attr: " << desired_layout_str;
      (*attrs)["data_format"] = desired_layout_str;
    } else if (attrs->find("data_layout") != attrs->end() &&
               PADDLE_GET_CONST(std::string, (*attrs)["data_layout"]) !=
                   desired_layout_str) {
      VLOG(4) << "Origin layout attr: "
              << PADDLE_GET_CONST(std::string, (*attrs)["data_layout"])
              << ", Desired layout attr: " << desired_layout_str;
      (*attrs)["data_layout"] = desired_layout_str;
    }

    // Step 2: Transpose the specified input for Op and set the transposed var's
    // layout.
    for (auto& name : this->Inputs()) {
      if (new_ins.find(name) != new_ins.end()) {
        auto& in_vars = new_ins[name];
        for (auto& var : in_vars) {
          if (var != nullptr &&
              paddle::imperative::GetDataLayout(var) != desired_layout) {
            var = TraceTransposeOp(var, desired_layout, tracer);
          }
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
class LightlyLayoutSensitiveOpTransformer : public LayoutTransformer<VarType> {
 public:
  explicit LightlyLayoutSensitiveOpTransformer(const std::string& type)
      : LayoutTransformer<VarType>(type) {}

  paddle::imperative::NameVarMap<VarType> Apply(
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
    auto desired_layout = LayoutAutoTune::Instance().GetDesiredLayout();
    for (auto& pair : new_ins) {
      for (auto& var : pair.second) {
        if (var != nullptr) {
          VLOG(3) << "Tune the layout from "
                  << paddle::framework::DataLayoutToString(
                         paddle::imperative::GetDataLayout(var))
                  << " to "
                  << paddle::framework::DataLayoutToString(
                         LayoutAutoTune::Instance().GetDesiredLayout());
        }
        if (var != nullptr &&
            paddle::imperative::GetDataLayout(var) == desired_layout &&
            desired_layout == DataLayout::NHWC) {
          // Set layout to UNDEFINED so that TransposeOpTransformer do
          // NHWC->NCHW transformation.
          var = TraceTransposeOp(var, DataLayout::UNDEFINED, tracer);
        }
      }
    }
    return new_ins;
  }
};

template <typename VarType>
class ElementwiseOpTransformer
    : public LightlyLayoutSensitiveOpTransformer<VarType> {
 public:
  explicit ElementwiseOpTransformer(const std::string& type)
      : LightlyLayoutSensitiveOpTransformer<VarType>(type) {}

  paddle::imperative::NameVarMap<VarType> Apply(
      const paddle::imperative::NameVarMap<VarType>& ins,
      const paddle::imperative::NameVarMap<VarType>& outs,
      paddle::framework::AttributeMap* attrs,
      const std::shared_ptr<paddle::imperative::Tracer>& tracer) {
    // [Why we need the this?]
    // The Elementwise Ops has a axis attr, it is to support broadcast.
    // When bias_attr of Conv is not false, the elementwise_add will be
    // appended, and the axis will be set to the channel dimension.
    // If the axis is set to the channel dimension, the attr transformation
    // is necessary. Otherwise, it will fall back to the
    // LayoutTransformer::Apply.
    auto& in1_vars = ins.at("X")[0];
    auto& in2_vars = ins.at("Y")[0];
    auto in_layout = paddle::imperative::GetDataLayout(in1_vars);
    // for conv's bias
    if (attrs->find("axis") != attrs->end() &&
        PADDLE_GET_CONST(int, (*attrs)["axis"]) != -1) {
      if (in_layout == DataLayout::NHWC) {
        (*attrs)["axis"] = 3;
      } else if (in_layout == DataLayout::NCHW) {
        (*attrs)["axis"] = 1;
      }
      this->SetVarsLayout(outs, in_layout);
      return ins;
    } else {
      auto in2_layout = paddle::imperative::GetDataLayout(in2_vars);
      if (in_layout == in2_layout) {
        this->SetVarsLayout(outs, in_layout);
        return ins;
      }
      return LightlyLayoutSensitiveOpTransformer<VarType>::Apply(
          ins, outs, attrs, tracer);
    }
  }
};

template <typename VarType>
class TransposeOpTransformer
    : public LightlyLayoutSensitiveOpTransformer<VarType> {
 public:
  explicit TransposeOpTransformer(const std::string& type)
      : LightlyLayoutSensitiveOpTransformer<VarType>(type) {}

  paddle::imperative::NameVarMap<VarType> Apply(
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
    auto desired_layout = LayoutAutoTune::Instance().GetDesiredLayout();
    if (var_layout == desired_layout && desired_layout == DataLayout::NHWC) {
      auto axis = PADDLE_GET_CONST(std::vector<int>, (*attrs)["axis"]);
      // NHWC->NCHW, permutaion will be set as follows.
      std::vector<int> perm = {0, 3, 1, 2};
      // fuse the transpose Ops by transforming axis.
      std::vector<int> fusion_axis = {
          perm[axis[0]], perm[axis[1]], perm[axis[2]], perm[axis[3]]};
      (*attrs)["axis"] = fusion_axis;
    }
    return ins;
  }
};

template <typename VarType>
class FlattenOpTransformer
    : public LightlyLayoutSensitiveOpTransformer<VarType> {
 public:
  explicit FlattenOpTransformer(const std::string& type)
      : LightlyLayoutSensitiveOpTransformer<VarType>(type) {}

  paddle::imperative::NameVarMap<VarType> Apply(
      const paddle::imperative::NameVarMap<VarType>& ins,
      const paddle::imperative::NameVarMap<VarType>& outs,
      paddle::framework::AttributeMap* attrs,
      const std::shared_ptr<paddle::imperative::Tracer>& tracer) {
    VLOG(3) << "Optimze lightly layout sensitive op " << this->Type();
    // Flatten the C, H, W dimensions will not affect functionality.
    // So transformation is unnecessary. But in other cases, it needs to
    // fall back to the LightlyLayoutSensitiveOpTransformer.
    auto start_axis = PADDLE_GET_CONST(int, (*attrs)["start_axis"]);
    auto stop_axis = PADDLE_GET_CONST(int, (*attrs)["stop_axis"]);
    if (paddle::imperative::GetDataLayout(ins.at("X")[0]) ==
            LayoutAutoTune::Instance().GetDesiredLayout() &&
        start_axis == 1 && stop_axis == 3) {
      return ins;
    } else {
      return LightlyLayoutSensitiveOpTransformer<VarType>::Apply(
          ins, outs, attrs, tracer);
    }
  }
};

template <typename VarType>
class ArgmaxOpTransformer
    : public LightlyLayoutSensitiveOpTransformer<VarType> {
 public:
  explicit ArgmaxOpTransformer(const std::string& type)
      : LightlyLayoutSensitiveOpTransformer<VarType>(type) {}

  paddle::imperative::NameVarMap<VarType> Apply(
      const paddle::imperative::NameVarMap<VarType>& ins,
      const paddle::imperative::NameVarMap<VarType>& outs,
      paddle::framework::AttributeMap* attrs,
      const std::shared_ptr<paddle::imperative::Tracer>& tracer) {
    VLOG(3) << "Optimze lightly layout sensitive op " << this->Type();
    auto& in_var = ins.at("X")[0];
    auto var_layout = paddle::imperative::GetDataLayout(in_var);
    bool keep_dims = PADDLE_GET_CONST(bool, (*attrs)["keepdims"]);
    if (keep_dims) {
      if (var_layout != DataLayout::UNDEFINED) {
        std::vector<int> perm_nhwc = {0, 3, 1, 2};
        std::vector<int> perm_nchw = {0, 2, 3, 1};

        auto perm = var_layout == DataLayout::NHWC ? perm_nhwc : perm_nchw;
        switch (AttrTypeID((*attrs)["axis"])) {
          case paddle::framework::proto::AttrType::INT: {
            auto axis = PADDLE_GET_CONST(int, (*attrs)["axis"]);
            (*attrs)["axis"] = static_cast<int>(perm[axis]);
          }
          case paddle::framework::proto::AttrType::LONG: {
            auto axis = PADDLE_GET_CONST(int64_t, (*attrs)["axis"]);
            (*attrs)["axis"] = static_cast<int64_t>(perm[axis]);
          }
          default:
            VLOG(4) << "The data_type of axis is Error, axis must be int or "
                       "int64, bug got "
                    << (AttrTypeID((*attrs)["axis"]));
        }
      }
      this->SetVarsLayout(outs, var_layout);
      return ins;
    }
    return LightlyLayoutSensitiveOpTransformer<VarType>::Apply(
        ins, outs, attrs, tracer);
  }
};

template <typename VarType>
class ConcatOpTransformer
    : public LightlyLayoutSensitiveOpTransformer<VarType> {
 public:
  explicit ConcatOpTransformer(const std::string& type)
      : LightlyLayoutSensitiveOpTransformer<VarType>(type) {}

  paddle::imperative::NameVarMap<VarType> Apply(
      const paddle::imperative::NameVarMap<VarType>& ins,
      const paddle::imperative::NameVarMap<VarType>& outs,
      paddle::framework::AttributeMap* attrs,
      const std::shared_ptr<paddle::imperative::Tracer>& tracer) {
    VLOG(3) << "Optimze lightly layout sensitive op " << this->Type();
    auto& in_var = ins.at("X")[0];
    auto var_layout = paddle::imperative::GetDataLayout(in_var);
    bool need_tranppose = false;
    for (auto& pair : ins) {
      for (auto& var : pair.second) {
        if (var != nullptr &&
            (paddle::imperative::GetDataLayout(var) != var_layout)) {
          need_tranppose = true;
          break;
        }
      }
    }

    if (need_tranppose) {
      return LightlyLayoutSensitiveOpTransformer<VarType>::Apply(
          ins, outs, attrs, tracer);
    }

    if (var_layout != DataLayout::UNDEFINED) {
      std::vector<int> perm_nhwc = {0, 3, 1, 2};
      std::vector<int> perm_nchw = {0, 2, 3, 1};
      auto perm = var_layout == DataLayout::NHWC ? perm_nhwc : perm_nchw;
      auto axis = PADDLE_GET_CONST(int, (*attrs)["axis"]);
      (*attrs)["axis"] = static_cast<int>(perm[axis]);
    }
    auto axis = PADDLE_GET_CONST(int, (*attrs)["axis"]);
    VLOG(3) << "Optimze lightly layout sensitive op asdfasdfasdf axis" << axis;

    this->SetVarsLayout(outs, var_layout);
    return ins;
  }
};

}  // namespace imperative
}  // namespace paddle
