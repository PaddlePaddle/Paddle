/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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
#include <vector>
#include "glog/logging.h"
#include "paddle/framework/ddim.h"
#include "paddle/framework/eigen.h"
#include "paddle/framework/operator.h"
#include "paddle/framework/tensor.h"
#include "paddle/operators/gather.h"
#include "paddle/operators/scatter.h"

namespace paddle {
namespace operators {

using namespace paddle::framework;

class CondOp : public OperatorBase {
 public:
  CondOp(const std::string& type, const VariableNameMap& inputs,
         const VariableNameMap& outputs, const AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {
    index_.resize(2);
    sub_net_op_.resize(2);
    LOG(INFO) << "Initialization Done.";
  }

  CondOp(const CondOp& o)
      : framework::OperatorBase(
            static_cast<const framework::OperatorBase&>(o)) {
    // TODO(yuyang18): Implement copy ctor well.
    PADDLE_THROW("Not implemented");
  }

  void CreateScope(const Scope& scope) const {
    auto sub_scopes_var = scope.FindVar("SubScopes");
    PADDLE_ENFORCE(sub_scopes_var != nullptr, "");
    auto sub_scopes = sub_scopes_var->GetMutable<std::vector<Scope*>>();
    auto& sub_scope = scope.NewScope();
    sub_scopes->push_back(&sub_scope);
  }

  void CreateIndexTensor(const Scope& scope) const {
    auto index_tensors_var = scope.FindVar("IndexTensors");
    PADDLE_ENFORCE(index_tensors_var != nullptr, "");
    auto& index_tensors =
        *index_tensors_var->GetMutable<std::vector<Tensor*>>();
    Tensor index_tensor;
    index_tensors.push_back(&index_tensor);
  }

  /**
   * InferShape must be called before Run.
   */
  void InferShape(const framework::Scope& scope) const override {
    auto sub_scopes_var = scope.FindVar("SubScopes");
    PADDLE_ENFORCE_NOT_NULL(sub_scopes_var);
    auto& sub_scopes = *sub_scopes_var->GetMutable<std::vector<Scope*>>();
    // auto& index_tensors =
    // *scope.FindVar("IndexTensors")->GetMutable<std::vector<Tensor*>>();

    for (int i = 0; i < 2; ++i) {
      // Create two sub scopes for true and false branches
      // sub_scopes[0] for the true branch and sub_scopes[1] for the false
      // branch
      CreateScope(scope);

      // Create two tensors for true and false indices
      // index_tensors[0] for the true branch and index_tensors[1] for the false
      // branch
      CreateIndexTensor(scope);

      for (auto& input : Inputs("Xs")) {
        // Create a new tensor in sub-scope for input-type tensor
        Variable* v = sub_scopes[i]->NewVar(input);
        Tensor* sub_input = v->GetMutable<Tensor>();
        sub_input->Resize(scope.FindVar(input)->GetMutable<Tensor>()->dims());
      }

      // Inputs that do not require tailoring
      /*for (auto& input : (*sub_net_op_[i]).Inputs()) {
        // weights are located in the parent scope rather than sub scope
        for (auto& var_name : input.second) {
          if (!sub_scopes[i]->FindVar(var_name)) {
            sub_scopes[i]->NewVar(var_name)->GetMutable<Tensor>();
          }
        }
      }*/

      // Outputs
      for (auto& output : (*sub_net_op_[i]).Outputs()) {
        for (auto& var_name : output.second) {
          sub_scopes[i]->NewVar(var_name);
        }
      }

      // each net calls InferShape
      LOG(INFO) << "OK 3";
      sub_net_op_[i]->InferShape(*sub_scopes[i]);
      LOG(INFO) << "OK 4";
    }

    for (auto& output : Outputs("Outs")) {
      Tensor* tensor_t_out =
          sub_scopes[0]->FindVar(output)->GetMutable<Tensor>();
      Tensor* tensor_f_out =
          sub_scopes[1]->FindVar(output)->GetMutable<Tensor>();
      Tensor* tensor_out = scope.FindVar(output)->GetMutable<Tensor>();
      // check output size should be same
      PADDLE_ENFORCE_EQ(tensor_t_out->dims(), tensor_f_out->dims(),
                        "Outputs not of the same shape");
      tensor_out->Resize(tensor_t_out->dims());
    }
    LOG(INFO) << "OK 5";
  }

  // Set True Block
  void set_truenet(std::unique_ptr<OperatorBase> net) {
    sub_net_op_[0] = std::move(net);
  }

  // Set False Block
  void set_falsenet(std::unique_ptr<OperatorBase> net) {
    sub_net_op_[1] = std::move(net);
  }

  void Run(const framework::Scope& scope,
           const platform::DeviceContext& dev_ctx) const override {
    auto sub_scopes = scope.FindVar("SubScopes")->Get<std::vector<Scope*>>();
    auto index_tensors =
        scope.FindVar("IndexTensors")->Get<std::vector<Tensor*>>();

    std::string cond_name = Input("Cond");
    Variable* cond_var = scope.FindVar(cond_name);
    PADDLE_ENFORCE_NOT_NULL(cond_var)
    const Tensor* cond = cond_var->GetMutable<Tensor>();

    // Step 1: get the true/false index at runtime
    // index_[0]: vector<int>, contains all index for cond[i] == true
    // index_[1]: vector<int>, contains all index for cond[i] == false
    for (int i = 0; i < 2; ++i) index_[i].clear();

    const bool* cond_data = cond->data<bool>();
    for (int i = 0; i < cond->dims()[0]; ++i) {
      if (cond_data[i])
        index_[0].push_back(i);
      else
        index_[1].push_back(i);
    }
    // put index_[0] and index_[1] into two tensors:
    // index_tensor_[0] and index_tensor_[1]
    framework::DDim dim = paddle::framework::make_ddim({0});
    for (int i = 0; i < 2; ++i) {
      dim[0] = index_[i].size();
      int* tmp_ptr =
          index_tensors[i]->mutable_data<int>(dim, platform::CPUPlace());
      index_tensors[i]->Resize(dim);
      memcpy(tmp_ptr, index_[i].data(), dim[0] * sizeof(int));
    }

    // Step 2: collect data by calling gather
    for (int i = 0; i < 2; ++i) {
      // i= 0/i for True and False branches respectively
      for (auto& input : Inputs("Xs")) {
        // find Tensor
        // Tensor* tensor_parent = scope.FindVar(input)->GetMutable<Tensor>();
        Variable* v = scope.FindVar(input);
        Tensor* tensor_parent = v->GetMutable<Tensor>();
        // Tensor* tensor_child =
        // sub_scope_[i].FindVar(input)->GetMutable<Tensor>();
        v = sub_scopes[i]->FindVar(input);
        Tensor* tensor_child = v->GetMutable<Tensor>();
        Gather<float>(dev_ctx.GetPlace(), tensor_parent, index_tensors[i],
                      tensor_child);
      }
    }

    // Step 3: run
    for (int i = 0; i < 2; ++i) sub_net_op_[i]->Run(*sub_scopes[i], dev_ctx);

    // Step 4: merge output results
    for (int i = 0; i < 2; ++i) {
      // i= 0/i for True and False branches respectively
      // for (auto& output : GetAttr<std::vector<std::string>>("sub_outputs")) {
      for (auto& output : Outputs("Outs")) {
        // find Tensor
        Variable* v = scope.FindVar(output);
        Tensor* tensor_parent = v->GetMutable<Tensor>();
        v = sub_scopes[i]->FindVar(output);
        Tensor* tensor_child = v->GetMutable<Tensor>();
        ScatterUpdate<float>(dev_ctx.GetPlace(), tensor_child, index_tensors[i],
                             tensor_parent);
      }
    }
  }

 private:
  // sub_net_op_[0]: subnet_t
  // sub_net_op_[1]: subnet_f
  std::vector<std::unique_ptr<framework::OperatorBase>> sub_net_op_;

  // index_[0]: True_index;
  // index_[1]: False_index;
  mutable std::vector<std::vector<int>> index_;
};

/*
class CondGradientOp final : public OperatorBase {
public:
        void Init() override;

        virtual void InferShape(const std::shared_ptr<Scope>& scope) const
override;

        virtual void Run(const std::shared_ptr<Scope>& scope,
                   const platform::DeviceContext& dev_ctx) const override;
};*/

}  // namespace operators
}  // namespace paddle
