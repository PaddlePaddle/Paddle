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

#include "paddle/operators/cond_op.h"

#include <cstring>
#include <sstream>

#include "paddle/framework/op_registry.h"
#include "paddle/operators/gather.h"
#include "paddle/operators/net_op.h"
#include "paddle/operators/scatter.h"

namespace paddle {
namespace operators {

using Scope = framework::Scope;
using Variable = framework::Variable;
using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
using DDim = framework::DDim;

void CondOp::CreateScope(const Scope& scope) const {
  auto sub_scopes_var = scope.FindVar("SubScopes");
  PADDLE_ENFORCE_NOT_NULL(sub_scopes_var,
                          "Output(SubScopes) of CondOp should not be null.");
  auto sub_scopes = sub_scopes_var->GetMutable<std::vector<Scope*>>();
  auto& sub_scope = scope.NewScope();
  sub_scopes->push_back(&sub_scope);
}

void CondOp::CreateIndexTensor(const Scope& scope) const {
  auto index_tensors_var = scope.FindVar("IndexTensors");
  PADDLE_ENFORCE_NOT_NULL(index_tensors_var,
                          "Output(IndexTensors) of CondOp should not be null.");
  auto& index_tensors =
      *index_tensors_var->GetMutable<std::vector<LoDTensor>>();
  index_tensors.push_back(LoDTensor());
}

void CondOp::InferShape(const Scope& scope) const {
  auto sub_scopes_var = scope.FindVar("SubScopes");
  PADDLE_ENFORCE_NOT_NULL(sub_scopes_var,
                          "Output(SubScopes) of CondOp should not be null.");
  auto& sub_scopes = *sub_scopes_var->GetMutable<std::vector<Scope*>>();

  for (int i = 0; i < 2; ++i) {
    // Create two sub scopes for true and false branches
    // sub_scopes[0] for the true branch and sub_scopes[1] for the false
    // branch
    CreateScope(scope);

    // Create two tensors for true and false indices
    // index_tensors[0] for the true branch and index_tensors[1] for the false
    // branch
    CreateIndexTensor(scope);

    PADDLE_ENFORCE(!Inputs("Xs").empty(),
                   "Inputs(Xs) of CondOp can't be empty.");
    for (auto& input : Inputs("Xs")) {
      // Create a new tensor in sub-scope for input-type tensor
      Variable* v = sub_scopes[i]->NewVar(input);
      LoDTensor* sub_input = v->GetMutable<LoDTensor>();
      sub_input->Resize(scope.FindVar(input)->GetMutable<LoDTensor>()->dims());
    }

    for (auto& output : (*sub_net_op_[i]).Outputs()) {
      for (auto& var_name : output.second) {
        sub_scopes[i]->NewVar(var_name);
      }
    }

    // each net calls InferShape
    sub_net_op_[i]->InferShape(*sub_scopes[i]);
  }

  for (auto& output : Outputs("Outs")) {
    LoDTensor* tensor_t_out =
        sub_scopes[0]->FindVar(output)->GetMutable<LoDTensor>();
    PADDLE_ENFORCE_NOT_NULL(tensor_t_out, "True output should not be NULL");
    LoDTensor* tensor_f_out =
        sub_scopes[1]->FindVar(output)->GetMutable<LoDTensor>();
    PADDLE_ENFORCE_NOT_NULL(tensor_f_out, "False output should not be NULL");

    auto* tensor_out_var = scope.FindVar(output);
    PADDLE_ENFORCE_NOT_NULL(tensor_out_var, "Output not found");
    LoDTensor* tensor_out = tensor_out_var->GetMutable<LoDTensor>();
    PADDLE_ENFORCE_NOT_NULL(tensor_t_out,
                            "True output tensor should not be NULL");

    // check output size should be same
    PADDLE_ENFORCE_EQ(tensor_t_out->dims(), tensor_f_out->dims(),
                      "Outputs not of the same shape");
    tensor_out->Resize(tensor_t_out->dims());
    // tensor_out->mutable_data<float>(tensor_out->dims(),
    // platform::CPUPlace());
    tensor_out->mutable_data<float>(platform::CPUPlace());
  }
}

void CondOp::Run(const Scope& scope,
                 const platform::DeviceContext& dev_ctx) const {
  auto* sub_scopes_var = scope.FindVar("SubScopes");
  PADDLE_ENFORCE_NOT_NULL(sub_scopes_var,
                          "Output(SubScopes) of CondOp should not be null.");
  auto sub_scopes = sub_scopes_var->Get<std::vector<Scope*>>();
  auto* index_tensors_var = scope.FindVar("IndexTensors");
  PADDLE_ENFORCE_NOT_NULL(index_tensors_var,
                          "Output(IndexTensors) of CondOp should not be null.");
  auto index_tensors = index_tensors_var->Get<std::vector<LoDTensor>>();

  std::string cond_name = Input("Cond");
  Variable* cond_var = scope.FindVar(cond_name);
  PADDLE_ENFORCE_NOT_NULL(cond_var,
                          "Input(Cond) of CondOp should not be null.");
  const LoDTensor* cond = cond_var->GetMutable<LoDTensor>();

  // Step 1: get the true/false index at runtime
  // index_[0]: vector<int>, contains all index for cond[i] == true
  // index_[1]: vector<int>, contains all index for cond[i] == false
  for (int i = 0; i < 2; ++i) index_[i].clear();

  const int* cond_data = cond->data<int>();
  for (int i = 0; i < cond->dims()[0]; ++i) {
    if (cond_data[i])
      index_[0].push_back(i);
    else
      index_[1].push_back(i);
  }

  // put index_[0] and index_[1] into two tensors:
  // index_tensor_[0] and index_tensor_[1]
  DDim dim = paddle::framework::make_ddim({0});
  for (int i = 0; i < 2; ++i) {
    dim[0] = index_[i].size();
    int* tmp_ptr =
        index_tensors[i].mutable_data<int>(dim, platform::CPUPlace());
    index_tensors[i].Resize(dim);
    memcpy(tmp_ptr, index_[i].data(), dim[0] * sizeof(int));
  }

  // Step 2: collect data by calling gather
  for (int i = 0; i < 2; ++i) {
    // i= 0/i for True and False branches respectively
    for (auto& input : Inputs("Xs")) {
      // find Tensor
      Variable* v = scope.FindVar(input);
      PADDLE_ENFORCE_NOT_NULL(v);
      LoDTensor* tensor_parent = v->GetMutable<LoDTensor>();

      v = sub_scopes[i]->FindVar(input);
      PADDLE_ENFORCE_NOT_NULL(v);
      LoDTensor* tensor_child = v->GetMutable<LoDTensor>();

      // Resize child
      DDim dim = tensor_child->dims();
      dim[0] = index_[i].size();
      tensor_child->Resize(dim);
      tensor_child->mutable_data<float>(dim, platform::CPUPlace());

      Gather<float>(dev_ctx.GetPlace(), tensor_parent, &index_tensors[i],
                    tensor_child);
    }
  }

  // Step 3: run
  for (int i = 0; i < 2; ++i) {
    sub_net_op_[i]->Run(*sub_scopes[i], dev_ctx);
  }

  // Step 4: merge output results
  PADDLE_ENFORCE(!Outputs("Outs").empty(),
                 "Outputs(Outs) of CondOp can't be empty.");
  for (int i = 0; i < 2; ++i) {
    // i= 0/i for True and False branches respectively
    for (auto& output : Outputs("Outs")) {
      // find Tensor
      Variable* v = scope.FindVar(output);
      PADDLE_ENFORCE_NOT_NULL(v);
      LoDTensor* tensor_parent = v->GetMutable<LoDTensor>();

      v = sub_scopes[i]->FindVar(output);
      PADDLE_ENFORCE_NOT_NULL(v);
      LoDTensor* tensor_child = v->GetMutable<LoDTensor>();

      ScatterUpdate<float>(dev_ctx.GetPlace(), tensor_child, &index_tensors[i],
                           tensor_parent);
    }
  }
}

class CondOpProtoAndCheckerMaker : public framework::OpProtoAndCheckerMaker {
 public:
  CondOpProtoAndCheckerMaker(framework::OpProto* proto,
                             framework::OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("Cond", "The condition, which is a bool vector");
    AddInput("Xs", "Inputs of Subnets").AsDuplicable();
    AddOutput("Outs", "Outputs of Cond_Op after merge").AsDuplicable();

    AddOutput("SubScopes", "sub scopes for true and false branches");
    AddOutput("IndexTensors", "Index Tensors contains indices for true/false");

    AddComment(R"DOC(
Sample dependent Cond Operator:
Given Cond[i] as a 1/0 vector to indicate true/false
The equation is: 
Out[i] = subnet_t[i], if Cond[i] == true
Out[i] = subnet_t[i], if Cond[i] == false
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP_WITHOUT_GRADIENT(cond, paddle::operators::CondOp,
                             paddle::operators::CondOpProtoAndCheckerMaker);
