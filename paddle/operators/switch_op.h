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
#include "glog/logging.h"
#include "paddle/framework/eigen.h"
#include "paddle/framework/operator.h"
#include "paddle/framework/ddim.h"
#include "paddle/operators/gather.h"

namespace paddle {
namespace operators {

using namespace paddle::framework;

template <typename Place, typename T>
class CondOp final : public OperatorBase {
public:
  void Init() override;

  /**
   * InferShape must be called before Run.
   */
  virtual void InferShape(const std::shared_ptr<Scope>& scope) const override {
    scope_t = scope.NewScope();
    scope_f = scope.NewScope();
    net_op_t->InferShape(scope_t);
    net_op_f->InferShape(scope_f);
    tensor_t = new Tensor();
    tensor_f = new Tensor();
    { // True branch
      for (auto& input : net_op_t->Inputs()) {
        auto var_name = input.second;
        if (!scope_t.FindVar(var_name) {
          scope_t.NewVar(var_name)->GetMutable<Tensor>();
        }
      }
    }
    { // False branch
      for (auto& input : net_op_f->Inputs()) {
        auto var_name = input.second;
        if (!scope_f.FindVar(var_name) {
          scope_f.NewVar(var_name)->GetMutable<Tensor>();
        }
      }
    }
  }

  virtual void Run(const std::shared_ptr<Scope>& scope,
                   const platform::DeviceContext& dev_ctx) const override {
    auto* cond = context.Input<Tensor>("Cond");
    // Step 1: get the index
    true_index.clear();
    false_index.clear();
    for(int i = 0; i < cond->dims()[0]; ++i) {
      if (cond->data<bool>()[i])
        true_index.push_back(i);
      else:
        false_index.push_back(i);
    }
    framework::DDim dim_ = paddle::framework::make_ddim({0});
    dim_[0] = true_index.size();
    tensor_t->Resize(dim_);
    // set value
    for (int i = 0; i < dim_[0]; ++i)
      tensor_t->mutable_data<int>()[i] = true_index[i];
    dim_[0] = false_index.size();
    tensor_f->Resize(dim_);
    // set value
    for (int i = 0; i < dim_[0]; ++i)
      tensor_f->mutable_data<int>()[i] = false_index[i];
    
    // Step 2: collect data by calling gather
    { // True branch
      for (auto& input : net_op_t->Inputs()) {
        auto var_name = input.second;
        // find Tensor
        Tensor* Tensor_parent = scope.FindVar(var_name)->GetMutable<Tensor>();
        Tensor* Tensor_child = scope_t.FindVar(var_name)->GetMutable<Tensor>();
        Gather<T>(dev_ctx.GetPlace(), tensor_parent, tensor_t, tensor_child); 
      }
      
    }
  }

private:
  Scope* scope_t;
  Scope* scope_f;

  // subnet_t
  std::unique_ptr<framework::OperatorBase> net_op_t;
  // NetOp* net_op_t;
  // subnet_f
  std::unique_ptr<framework::OperatorBase> net_op_f;
  // NetOp* net_op_f;

  // T_index
  vector<int> true_index;
  Tensor* tensor_t;
  // F_index
  vector<int> false_index;
  Tensor* tensor_f;
};

class CondOpMaker : public OpProtoAndCheckerMaker {
public:
  IfElseOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("Cond", "The condition, which is a bool vector");
    AddAttr<std::string>("subnet_t", "The subnet network to be called when Cond[i] == true");
    AddAttr<std::string>("subnet_f", "The subnet network to be called when Cond[i] == false");
    AddOutput("Out", "The output of if-else op");
    AddComment(R"DOC(
Sample dependent Cond Operator:
The equation is: Out[i] = subnet_t[i], if Cond[i] == true
Out[i] = subnet_t[i], if Cond[i] == false
)DOC");
  }
};

class CondGradientOp final : public OperatorBase {
public:
	void Init() override;

	virtual void InferShape(const std::shared_ptr<Scope>& scope) const override;

	virtual void Run(const std::shared_ptr<Scope>& scope,
                   const platform::DeviceContext& dev_ctx) const override;
};

}  // namespace operators
}  // namespace paddle
