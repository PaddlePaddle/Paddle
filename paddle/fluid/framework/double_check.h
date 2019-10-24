/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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
#include <map>
#include <string>
#include <vector>

#include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace framework {

class DoubleCheckOperator {
 public:
  explicit DoubleCheckOperator(const OperatorBase& base_op)
      : base_op_(base_op) {}
  void Run(const Scope& scope, const platform::Place& place) {
    std::string type = base_op_.Type();
    if (type == "cast") {
      return;
    }

    // AttributeMap attrs;
    if (type == "dropout") {
      AttributeMap attrs = base_op_.Attrs();
      attrs["is_test"] = true;
      // set random seed in attrs.
      auto dropout_op = paddle::framework::OpRegistry::CreateOp(
          type, base_op_.Inputs(), base_op_.Outputs(), attrs);
      dropout_op->Run(scope, place);
    }

    VariableNameMap inputs;
    VariableNameMap outputs;

    std::map<std::string, std::string> input_diff_var_names;
    std::map<std::string, std::string> output_diff_var_names;

    Scope* var_scope = const_cast<Scope*>(&scope);
    PrepareNameMap(var_scope, place, base_op_.Inputs(), &inputs,
                   &input_diff_var_names);
    PrepareNameMap(var_scope, place, base_op_.Outputs(), &outputs,
                   &output_diff_var_names);

    if (input_diff_var_names.size() == 0 && output_diff_var_names.size() == 0) {
      VLOG(10) << base_op_.Type() << " no fp16 should be checked";
      return;
    }

    VLOG(10) << "begin to check " << base_op_.Type();
    auto check_op = paddle::framework::OpRegistry::CreateOp(
        type, inputs, outputs, base_op_.Attrs());
    check_op->Run(scope, place);

    for (auto it : output_diff_var_names) {
      VLOG(10) << "begin to var_name: " << it.first << " and " << it.second;
      Diff(scope, place, it.first, it.second);
    }
  }

 private:
  struct RangeFunctor {
    RangeFunctor(const platform::float16* a, const float* b) : a_(a), b_(b) {}
    inline HOSTDEVICE void operator()(size_t id) const {
      PADDLE_ENFORCE((fabs(static_cast<float>(a_[id]) - b_[id]) < 0.000001));
    }
    const platform::float16* a_;
    const float* b_;
  };

  void Diff(const Scope& scope, const platform::Place& place,
            const std::string& a, const std::string& b) {
    auto var_a = scope.FindVar(a);
    const Tensor* tensor_a{nullptr};
    if (var_a->IsType<framework::LoDTensor>()) {
      tensor_a = &var_a->Get<framework::LoDTensor>();
    } else if (var_a->IsType<framework::SelectedRows>()) {
      tensor_a = &var_a->Get<framework::SelectedRows>().value();
    }

    auto var_b = scope.FindVar(b);
    const Tensor* tensor_b{nullptr};
    if (var_a->IsType<framework::LoDTensor>()) {
      tensor_b = &var_b->Get<framework::LoDTensor>();
    } else if (var_a->IsType<framework::SelectedRows>()) {
      tensor_b = &var_b->Get<framework::SelectedRows>().value();
    }

    auto* dev_ctx = platform::DeviceContextPool::Instance().Get(place);
    RangeFunctor functor(tensor_a->data<platform::float16>(),
                         tensor_b->data<float>());
    platform::ForRange<platform::DeviceContext> for_range(*dev_ctx,
                                                          tensor_a->numel());
  }

  void PrepareNameMap(Scope* scope, const platform::Place& place,
                      const VariableNameMap& name_map,
                      VariableNameMap* dst_name_map,
                      std::map<std::string, std::string>* diff_var_names) {
    for (auto it = name_map.begin(); it != name_map.end();) {
      std::vector<std::string>& dst_var_names = (*dst_name_map)[it->first];
      dst_var_names.reserve(it->second.size());

      auto& var_names = it->second;
      for (size_t i = 0; i < var_names.size(); ++i) {
        auto var_name = var_names[i];
        auto var = scope->FindVar(var_name);
        std::string fp32_var_name = var_name + "_fp32_double_check_fake_name";

        const framework::Tensor* tensor{nullptr};
        if (var->IsType<framework::LoDTensor>()) {
          tensor = &var->Get<framework::LoDTensor>();
        } else if (var->IsType<framework::SelectedRows>()) {
          tensor = &var->Get<framework::SelectedRows>().value();
        } else {
          dst_var_names.push_back(fp32_var_name);
          continue;
        }

        auto tensor_dtype = tensor->type();
        if (tensor_dtype != framework::proto::VarType::FP16) {
          dst_var_names.push_back(fp32_var_name);
          continue;
        }

        dst_var_names.push_back(fp32_var_name);
        (*diff_var_names)[var_name] = fp32_var_name;

        if (scope->FindVar(fp32_var_name) != nullptr) {
          continue;
        }

        auto fp32_var = scope->Var(fp32_var_name);
        framework::Tensor* fp32_tensor{nullptr};
        if (var->IsType<framework::LoDTensor>()) {
          fp32_tensor = fp32_var->GetMutable<framework::LoDTensor>();
        } else if (var->IsType<framework::SelectedRows>()) {
          fp32_tensor =
              fp32_var->GetMutable<framework::SelectedRows>()->mutable_value();
        }
        fp32_tensor->mutable_data(place, tensor_dtype, tensor->memory_size());

        framework::AttributeMap cast_op_attrs;
        cast_op_attrs["in_dtype"] = framework::proto::VarType::FP16;
        cast_op_attrs["out_dtype"] = framework::proto::VarType::FP32;
        auto cast_op = paddle::framework::OpRegistry::CreateOp(
            "cast", {{"X", {var_name}}}, {{"Out", {fp32_var_name}}},
            cast_op_attrs);
        cast_op->Run(*scope, place);
      }
    }

    return;
  }

 private:
  const OperatorBase& base_op_;
};
}  // namespace framework
}  // namespace paddle
