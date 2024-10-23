/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <string>

#include "paddle/fluid/framework/custom_operator_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/phi/api/ext/op_meta_info.h"

namespace paddle {
namespace framework {

class CustomOpMaker : public OpProtoAndCheckerMaker {
 public:
  explicit CustomOpMaker(const std::vector<std::string>& inputs,
                         const std::vector<std::string>& outputs,
                         const std::vector<std::string>& attrs)
      : inputs_(inputs), outputs_(outputs), attrs_(attrs) {}

  void Make() override {
    for (auto& in_name : inputs_) {
      auto input_var_builder =
          AddInput(in_name, "The input " + in_name + "of Custom operator.");
      if (detail::IsDuplicableVar(in_name)) {
        input_var_builder.AsDuplicable();
      }
      if (detail::IsOptionalVar(in_name)) {
        input_var_builder.AsDispensable();
      }
    }
    for (auto& out_name : outputs_) {
      auto output_var_builder =
          AddOutput(out_name, "The output " + out_name + "of Custom Operator.");
      if (detail::IsDuplicableVar(out_name)) {
        output_var_builder.AsDuplicable();
      }
      if (detail::IsOptionalVar(out_name)) {
        output_var_builder.AsDispensable();
      }
    }
    for (auto& attr : attrs_) {
      auto attr_name_and_type = paddle::ParseAttrStr(attr);
      auto attr_name = attr_name_and_type[0];
      auto attr_type_str = attr_name_and_type[1];
      if (attr_type_str == "bool") {
        AddAttr<bool>(attr_name, "custom operator bool attribute.")
            .SetDefault(false);
      } else if (attr_type_str == "int") {
        AddAttr<int>(attr_name, "custom operator int attribute.").SetDefault(1);
      } else if (attr_type_str == "float") {
        AddAttr<float>(attr_name, "custom operator float attribute.")
            .SetDefault(1.0f);
      } else if (attr_type_str == "double") {
        AddAttr<double>(attr_name, "custom operator double attribute.")
            .SetDefault(1.0f);
      } else if (attr_type_str == "int64_t") {
        AddAttr<int64_t>(attr_name, "custom operator int64_t attribute.")
            .SetDefault(1);
      } else if (attr_type_str == "std::string") {
        AddAttr<std::string>(attr_name, "custom operator int attribute.")
            .SetDefault("");
      } else if (attr_type_str == "std::vector<int>") {
        AddAttr<std::vector<int>>(attr_name,
                                  "custom operator std::vector<int> attribute.")
            .SetDefault({});
      } else if (attr_type_str == "std::vector<float>") {
        AddAttr<std::vector<float>>(
            attr_name, "custom operator std::vector<float> attribute.")
            .SetDefault({});
      } else if (attr_type_str == "std::vector<int64_t>") {
        AddAttr<std::vector<int64_t>>(
            attr_name, "custom operator std::vector<int64_t> attribute.")
            .SetDefault({});
      } else if (attr_type_str == "std::vector<std::string>") {
        AddAttr<std::vector<std::string>>(
            attr_name, "custom operator std::vector<std::string> attribute.")
            .SetDefault({});
      } else {
        PADDLE_THROW(common::errors::Unimplemented(
            "Unsupported `%s` type value as custom attribute now. "
            "Supported data types include `bool`, `int`, `float`, `double`"
            "`int64_t`, `std::string`, `std::vector<int>`, "
            "`std::vector<float>`, `std::vector<int64_t>`, "
            "`std::vector<std::string>`, Please check whether "
            "the attribute data type and data type string are matched.",
            attr_type_str));
      }
    }
    AddComment(R"DOC(
Custom Operator.

According to the phi::DenseTensor operation function implemented by the user
independently of the framework, it is encapsulated into a framework
operator to adapt to various execution scenarios such as dynamic graph
mode, static graph mode, and inference mode.

)DOC");
  }

 private:
  std::vector<std::string> inputs_;
  std::vector<std::string> outputs_;
  std::vector<std::string> attrs_;
};

template <typename T>
class CustomGradOpMaker;

template <>
class CustomGradOpMaker<OpDesc> : public SingleGradOpMaker<OpDesc> {
 public:
  explicit CustomGradOpMaker(
      const OpDesc& fwd_op,
      const std::unordered_set<std::string>& no_grad_set,
      std::unordered_map<std::string, std::string>* grad_to_var,
      const std::vector<BlockDesc*>& grad_block,
      const std::string& name,
      const std::vector<std::string>& inputs,
      const std::vector<std::string>& outputs,
      bool is_double_grad)
      : SingleGradOpMaker<OpDesc>(fwd_op, no_grad_set, grad_to_var, grad_block),
        name_(name),
        inputs_(inputs),
        outputs_(outputs),
        is_double_grad_(is_double_grad) {}

 protected:
  void Apply(GradOpPtr<OpDesc> grad_op) const override {
    grad_op->SetType(name_);

    auto fwd_op_inputs = this->InputNames();
    auto fwd_op_outputs = this->OutputNames();

    for (auto& in_name : inputs_) {
      VLOG(3) << "Custom Operator: GradOpDescMaker - input: " << in_name;
      if (!detail::IsGradVar(in_name, is_double_grad_)) {
        if (detail::IsMemberOf(fwd_op_inputs, in_name)) {
          grad_op->SetInput(in_name, this->Input(in_name));
        } else if (detail::IsMemberOf(fwd_op_outputs, in_name)) {
          grad_op->SetInput(in_name, this->Output(in_name));
        } else {
          PADDLE_THROW(common::errors::InvalidArgument(
              "The input tensor name `%s` is invalid, expected it is the input "
              "or output of forward operator.",
              in_name));
        }
      } else {
        if (this->HasOutput(detail::NoGrad(in_name))) {
          grad_op->SetInput(in_name, this->OutputGrad(detail::NoGrad(in_name)));
        } else {
          // Maybe visit here! handle inplace optional case
          PADDLE_ENFORCE(
              in_name.find(paddle::kOptionalSuffix) != std::string::npos,
              common::errors::InvalidArgument(
                  "Custom operator couldn't find grad operator input name for "
                  "%s. If you are using inplace optional inputs & outputs, "
                  "please check your InplaceMap and `Outputs` again and make "
                  "sure %s is wrapped by `paddle::Optional`",
                  in_name,
                  in_name));
          VLOG(3) << "Custom Operator: GradOpDescMaker - handle unfound input: "
                  << in_name;
        }
      }
    }
    for (auto& out_name : outputs_) {
      // Handle inplace optional case
      if (!this->HasInput(detail::NoGrad(out_name, is_double_grad_))) {
        PADDLE_ENFORCE(
            out_name.find(paddle::kOptionalSuffix) != std::string::npos,
            common::errors::InvalidArgument(
                "Custom operator couldn't find grad operator output name for "
                "%s. If you are using inplace optional inputs & outputs, "
                "please check your InplaceMap and `Outputs` again and make "
                "sure %s is wrapped by `paddle::Optional`",
                out_name,
                out_name));
        VLOG(3) << "Custom Operator: GradOpDescMaker - handle unfound output: "
                << out_name;
        continue;
      }
      VLOG(3) << "Custom Operator: GradOpDescMaker - output: " << out_name;
      if (detail::IsDuplicableVar(out_name)) {
        grad_op->SetOutput(
            out_name,
            this->InputGrad(detail::NoGrad(out_name, is_double_grad_),
                            /*drop_empty_grad=*/false));
      } else {
        grad_op->SetOutput(
            out_name,
            this->InputGrad(detail::NoGrad(out_name, is_double_grad_)));
      }
    }
    grad_op->SetAttrMap(this->Attrs());
  }

 private:
  std::string name_;
  std::vector<std::string> inputs_;
  std::vector<std::string> outputs_;
  bool is_double_grad_{false};
};

template <>
class CustomGradOpMaker<imperative::OpBase>
    : public SingleGradOpMaker<imperative::OpBase> {
 public:
  explicit CustomGradOpMaker(
      const std::string& type,
      const imperative::NameVarBaseMap& var_base_map_in,
      const imperative::NameVarBaseMap& var_base_map_out,
      const AttributeMap& attrs,
      const std::map<std::string, std::string>& inplace_map,
      const std::string& name,
      const std::vector<std::string>& inputs,
      const std::vector<std::string>& outputs,
      bool is_double_grad)
      : SingleGradOpMaker<imperative::OpBase>(
            type, var_base_map_in, var_base_map_out, attrs, inplace_map),
        name_(name),
        inputs_(inputs),
        outputs_(outputs),
        is_double_grad_(is_double_grad) {}

 protected:
  // TODO(chenweihang): The code is duplicated with the previous one, because
  // ere OpMaker's Input, Output and other methods are protected. Putting the
  // function implementation outside the class will cause the method to be
  // uncallable,
  // so it is still implemented in the class for the time being.
  void Apply(GradOpPtr<imperative::OpBase> grad_op) const override {
    grad_op->SetType(name_);

    auto fwd_op_inputs = this->InputNames();
    auto fwd_op_outputs = this->OutputNames();

    for (auto& in_name : inputs_) {
      VLOG(3) << "Custom Operator: GradOpBaseMaker - input: " << in_name;
      if (!detail::IsGradVar(in_name, is_double_grad_)) {
        if (detail::IsMemberOf(fwd_op_inputs, in_name)) {
          grad_op->SetInput(in_name, this->Input(in_name));
        } else if (detail::IsMemberOf(fwd_op_outputs, in_name)) {
          grad_op->SetInput(in_name, this->Output(in_name));
        } else {
          PADDLE_THROW(common::errors::InvalidArgument(
              "The input tensor name `%s` is invalid, expected it is the input "
              "or output of forward operator.",
              in_name));
        }
      } else {
        // Handle inplace optional case
        if (this->HasOutput(detail::NoGrad(in_name))) {
          grad_op->SetInput(in_name, this->OutputGrad(detail::NoGrad(in_name)));
        } else {
          PADDLE_ENFORCE(
              in_name.find(paddle::kOptionalSuffix) != std::string::npos,
              common::errors::InvalidArgument(
                  "Custom operator couldn't find grad operator input name for "
                  "%s. If you are using inplace optional inputs & outputs, "
                  "please check your InplaceMap and `Outputs` again and make "
                  "sure %s is wrapped by `paddle::Optional`",
                  in_name,
                  in_name));
          VLOG(3) << "Custom Operator: GradOpBaseMaker - handle unfound input: "
                  << in_name;
        }
      }
    }
    for (auto& out_name : outputs_) {
      // Handle inplace optional case
      if (!this->HasInput(detail::NoGrad(out_name, is_double_grad_))) {
        PADDLE_ENFORCE(
            out_name.find(paddle::kOptionalSuffix) != std::string::npos,
            common::errors::InvalidArgument(
                "Custom operator couldn't find grad operator output name for "
                "%s. If you are using inplace optional inputs & outputs, "
                "please check your InplaceMap and `Outputs` again and make "
                "sure %s is wrapped by `paddle::Optional`",
                out_name,
                out_name));
        VLOG(3) << "Custom Operator: GradOpBaseMaker - handle unfound output: "
                << out_name;
        continue;
      }
      VLOG(3) << "Custom Operator: GradOpBaseMaker - output: " << out_name;
      grad_op->SetOutput(
          out_name, this->InputGrad(detail::NoGrad(out_name, is_double_grad_)));
    }
    grad_op->SetAttrMap(this->Attrs());
  }

 private:
  std::string name_;
  std::vector<std::string> inputs_;
  std::vector<std::string> outputs_;
  bool is_double_grad_{false};
};

// Load custom op api: register op after user compiled
const std::unordered_map<std::string, std::vector<OpMetaInfo>>&
LoadOpMetaInfoAndRegisterOp(const std::string& dso_name);

// Register custom op api: register op directly
void RegisterOperatorWithMetaInfoMap(
    const paddle::OpMetaInfoMap& op_meta_info_map, void* dso_handle = nullptr);

// Interface for selective register custom op.
void RegisterOperatorWithMetaInfo(const std::vector<OpMetaInfo>& op_meta_infos,
                                  void* dso_handle = nullptr);
}  // namespace framework
}  // namespace paddle
