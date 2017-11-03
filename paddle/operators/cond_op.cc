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

#include "paddle/operators/gather.h"
#include "paddle/operators/scatter.h"

namespace paddle {
namespace operators {

using Scope = framework::Scope;
using Variable = framework::Variable;
using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
using DDim = framework::DDim;

framework::Scope& CondOp::AddSubScope(const Scope& scope) const {
  auto sub_scopes_var = scope.FindVar("SubScopes");
  PADDLE_ENFORCE_NOT_NULL(sub_scopes_var,
                          "Output(SubScopes) of CondOp should not be null.");
  auto sub_scopes = sub_scopes_var->GetMutable<std::vector<Scope*>>();
  auto& sub_scope = scope.NewScope();
  sub_scopes->push_back(&sub_scope);
  return sub_scope;
}

std::vector<framework::Scope*>& CondOp::GetSubScopes(
    const framework::Scope& scope) const {
  auto sub_scopes_var = scope.FindVar("SubScopes");
  PADDLE_ENFORCE_NOT_NULL(sub_scopes_var,
                          "Output(SubScopes) of CondOp should not be null.");
  return *sub_scopes_var->GetMutable<std::vector<framework::Scope*>>();
}

LoDTensor& CondOp::AddIndexTensor(const Scope& scope) const {
  auto index_tensors_var = scope.FindVar("IndexTensors");
  PADDLE_ENFORCE_NOT_NULL(index_tensors_var,
                          "Output(IndexTensors) of CondOp should not be null.");
  auto& index_tensors =
      *index_tensors_var->GetMutable<std::vector<LoDTensor>>();
  index_tensors.push_back(LoDTensor());
  return index_tensors.back();
}

std::vector<framework::LoDTensor>& CondOp::GetIndexTensors(
    const framework::Scope& scope) const {
  auto* index_tensors_var = scope.FindVar("IndexTensors");
  PADDLE_ENFORCE_NOT_NULL(index_tensors_var,
                          "Output(IndexTensors) of CondOp should not be null.");
  return *index_tensors_var->GetMutable<std::vector<framework::LoDTensor>>();
}

void CondOp::PrepareDataForSubnet(
    const framework::Scope& scope,
    const platform::DeviceContext& dev_ctx) const {
  PADDLE_ENFORCE(!Inputs("Xs").empty(), "Inputs(Xs) of CondOp can't be empty.");

  for (int i = 0; i < BRANCH_NUM; ++i) {
    // Create two sub scopes for true and false branches
    //   sub_scopes[0] for the true branch
    //   sub_scopes[1] for the false branch
    AddSubScope(scope);
    // Create two tensors for true and false indices:
    //   index_tensors[0] for the true branch
    //   index_tensors[1] for the false branch
    AddIndexTensor(scope);
  }

  Variable* cond_var = scope.FindVar(Input("Cond"));
  PADDLE_ENFORCE_NOT_NULL(cond_var,
                          "Input(Cond) of CondOp should not be null.");
  const LoDTensor* cond = cond_var->GetMutable<LoDTensor>();

  // get the true/false index at runtime according to cond tensor
  // index_vectors[0]: vector<int>, contains all index for cond[i] == true
  // index_vectors[1]: vector<int>, contains all index for cond[i] == false
  std::vector<std::vector<int>> index_vectors;
  index_vectors.resize(BRANCH_NUM);

  const int* cond_data = cond->data<int>();
  for (int i = 0; i < cond->dims()[0]; ++i) {
    if (cond_data[i])
      index_vectors[TRUE_BRANCH].push_back(i);
    else
      index_vectors[FALSE_BRANCH].push_back(i);
  }

  // put index_vectors[0] and index_vectors[1] into two tensors:
  // index_tensors[0] and index_tensors[1]
  std::vector<framework::LoDTensor>& index_tensors = GetIndexTensors(scope);
  std::vector<framework::Scope*>& sub_scopes = GetSubScopes(scope);

  for (int i = 0; i < BRANCH_NUM; ++i) {
    DDim dim = {static_cast<int64_t>(index_vectors[i].size())};
    int* index_tensor_data_ptr =
        index_tensors[i].mutable_data<int>(dim, platform::CPUPlace());
    memcpy(index_tensor_data_ptr, index_vectors[i].data(),
           dim[0] * sizeof(int));
  }

  // create input in subscopes according to index_vectors
  for (auto& input : Inputs("Xs")) {
    Variable* var_parent = scope.FindVar(input);
    PADDLE_ENFORCE_NOT_NULL(var_parent);
    const auto* tensor_parent = &var_parent->Get<LoDTensor>();

    for (int i = 0; i < BRANCH_NUM; ++i) {
      Variable* var_child = sub_scopes[i]->FindVar(input);
      PADDLE_ENFORCE_NOT_NULL(var_child);
      auto* tensor_child = var_child->GetMutable<LoDTensor>();

      // Resize child
      DDim dim = tensor_parent->dims();
      dim[0] = index_tensors[i].dims()[0];
      tensor_child->mutable_data<float>(dim, platform::CPUPlace());

      CPUGather<float>(dev_ctx, *tensor_parent, index_tensors[i], tensor_child);
    }
  }

  // create output_tensors in subscope for sub_net
  for (int i = 0; i < BRANCH_NUM; ++i) {
    for (auto& output : (*sub_net_op_[i]).Outputs()) {
      for (auto& var_name : output.second) {
        sub_scopes[i]->Var(var_name);
      }
    }
  }
}

void CondOp::MergeDataFromSubnet(const framework::Scope& scope,
                                 const platform::DeviceContext& dev_ctx) const {
  std::vector<framework::Scope*>& sub_scopes = GetSubScopes(scope);
  const std::vector<framework::LoDTensor>& index_tensors =
      GetIndexTensors(scope);

  // Infer the output dim, out_dim[0] = true_dim[0] + false_dim[0]
  PADDLE_ENFORCE(!Outputs("Outs").empty(),
                 "Outputs(Outs) of CondOp can't be empty.");
  for (auto& output : Outputs("Outs")) {
    const LoDTensor* tensor_t_out =
        &sub_scopes[TRUE_BRANCH]->FindVar(output)->Get<LoDTensor>();
    PADDLE_ENFORCE_NOT_NULL(tensor_t_out, "True output should not be NULL");
    const LoDTensor* tensor_f_out =
        &sub_scopes[FALSE_BRANCH]->FindVar(output)->Get<LoDTensor>();
    PADDLE_ENFORCE_NOT_NULL(tensor_f_out, "False output should not be NULL");

    auto* var_out = scope.FindVar(output);
    PADDLE_ENFORCE_NOT_NULL(var_out, "Output not found");
    LoDTensor* tensor_out = var_out->GetMutable<LoDTensor>();
    PADDLE_ENFORCE_NOT_NULL(tensor_t_out,
                            "True output tensor should not be NULL");

    DDim true_dim = tensor_t_out->dims();
    DDim false_dim = tensor_f_out->dims();
    true_dim[0] = 0;
    false_dim[0] = 0;
    PADDLE_ENFORCE_EQ(true_dim, false_dim,
                      "Outputs not of the same shape except the first dim");

    DDim out_dim = tensor_t_out->dims();
    out_dim[0] = tensor_t_out->dims()[0] + tensor_f_out->dims()[0];
    tensor_out->Resize(out_dim);
    tensor_out->mutable_data<float>(platform::CPUPlace());
  }

  // merge output results:
  // output_tensor = true_output_tensor + false_output_tensor
  for (auto& output : Outputs("Outs")) {
    Variable* var_parent = scope.FindVar(output);
    PADDLE_ENFORCE_NOT_NULL(var_parent);
    auto* tensor_parent = var_parent->GetMutable<LoDTensor>();

    for (int i = 0; i < BRANCH_NUM; ++i) {
      Variable* var_child = sub_scopes[i]->FindVar(output);
      PADDLE_ENFORCE_NOT_NULL(var_child);
      auto* tensor_child = &var_child->Get<LoDTensor>();
      ScatterAssign<float>(dev_ctx, *tensor_child, index_tensors[i],
                           tensor_parent);
    }
  }
}

void CondOp::Run(const Scope& scope,
                 const platform::DeviceContext& dev_ctx) const {
  PrepareDataForSubnet(scope, dev_ctx);
  std::vector<framework::Scope*>& sub_scopes = GetSubScopes(scope);
  for (int i = 0; i < BRANCH_NUM; ++i) {
    sub_net_op_[i]->Run(*sub_scopes[i], dev_ctx);
  }
  MergeDataFromSubnet(scope, dev_ctx);
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
Sample Dependent Conditional Operator.

Given Cond[i] as a 1/0 vector to indicate true/false:
Out[i] = subnet_true[i], if Cond[i] == true
Out[i] = subnet_false[i], if Cond[i] == false

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP_WITHOUT_GRADIENT(cond, paddle::operators::CondOp,
                             paddle::operators::CondOpProtoAndCheckerMaker);
