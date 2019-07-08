// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/filter_instag_op.h"

#include <memory>
#include "paddle/fluid/framework/no_need_buffer_vars_inference.h"
#include "paddle/fluid/framework/var_type_inference.h"

namespace paddle {
namespace operators {
class FilterInstagOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X1"), "Input(X1) should be not null.");
    PADDLE_ENFORCE(ctx->HasInput("X2"), "Input(X2) should be not null.");
    PADDLE_ENFORCE(ctx->HasInput("X3"), "Input(X3) should be not null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"), "Output(Out) should be not null.");
    PADDLE_ENFORCE(ctx->HasOutput("Map"), "Output(Map) should be not null.");

    if (!ctx->IsRuntime()) {
      auto x1_dims = ctx->GetInputDim("X1");  // batch_size * vec

      ctx->SetOutputDim("Out", framework::make_ddim({-1, x1_dims[1]}));
      ctx->SetOutputDim("Map", framework::make_ddim({-1, 2}));

    } else {
      auto x1_dims = ctx->GetInputDim("X1");
      ctx->SetOutputDim("Out", framework::make_ddim({x1_dims[0], x1_dims[1]}));
      ctx->SetOutputDim("Map", framework::make_ddim({x1_dims[0], 2}));
      /*auto x1 = Scope.Var("X1")->GetMutable<LoDTensor>();
      auto x2 = Scope.Var("X2")->GetMutable<LoDTensor>();
      auto x3 = Scope.Var("X3")->GetMutable<Tensor>();
      template <typename T>;
      using Vector = framework::CPUVector<T>;
      std::unordered_set<int64_t> filter_tag;
      auto* x3_data = x3->data<int64_t>();
      for (size_t i = 0; i < len; i++) {
          filter_tag.insert(x3_data[i]);
      }
      auto* x2_data = x2->data<int64_t>();
      auto x2_lods = x2->lod()[0];
      auto x1_lods = x1->lod()[0];
      std::vector<size_t> ins_after_filter;
      Vector<size_t> out_lods(1, 0);
      for (size_t i = 0; i < x2_lods.size() -1 ; i++) {
          for(size_t j = x2_lods[i]; j < x2_lods[i+1]; j++) {
              if (filter_tag.find(x2_data[j]) != filter_tag.end()) {
      */
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = framework::GetDataTypeOfVar(ctx.InputVar("X1"));
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};

class FilterInstagOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X1", "(LoDTensor) global fc output");
    AddInput("X2", "(LoDTensor) ins tag list");
    AddInput("X3", "(1D Tensor) fc tag list");
    AddOutput("Out", "(LoDTensor) global fc split to local fc");
    AddOutput("Map", "(LoDTensor) mapping from Out rows to X1 rows");
    AddComment(R"DOC(
Lookup Table Operator.
                
This operator is used to perform lookups on the parameter W,
then concatenated into a dense tensor.
                         
The input Ids can carry the LoD (Level of Details) information,
or not. And the output only shares the LoD information with input Ids.                         
)DOC");
  }
};

class FilterInstagOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Map"), "Input(Map) should be not null");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Grad Input(Out) should be not null");
    PADDLE_ENFORCE(ctx->HasInput("X1"), "Input(X1) should be not null");
    PADDLE_ENFORCE(ctx->HasOutput(framework::GradVarName("X1")),
                   "Grad Output(X1) should be not null");
    auto grad_out_dims = ctx->GetInputDim(framework::GradVarName("Out"));
    auto x1_dims = ctx->GetInputDim("X1");
    ctx->SetOutputDim(framework::GradVarName("X1"),
                      framework::make_ddim({x1_dims[0], grad_out_dims[1]}));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = framework::GetDataTypeOfVar(
        ctx.InputVar(framework::GradVarName("Out")));
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};

class FilterInstagGradOpDescMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    std::unique_ptr<framework::OpDesc> op(new framework::OpDesc());
    op->SetType("filter_instag_grad");
    op->SetInput("Map", Output("Map"));
    op->SetInput("X1", Input("X1"));
    op->SetInput(framework::GradVarName("Out"), OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X1"), InputGrad("X1"));
    return op;
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(filter_instag, ops::FilterInstagOp, ops::FilterInstagOpMaker,
                  ops::FilterInstagGradOpDescMaker);

REGISTER_OPERATOR(filter_instag_grad, ops::FilterInstagOpGrad);

REGISTER_OP_CPU_KERNEL(filter_instag, ops::FilterInstagKernel<float>,
                       ops::FilterInstagKernel<double>,
                       ops::FilterInstagKernel<int32_t>,
                       ops::FilterInstagKernel<int64_t>);

REGISTER_OP_CPU_KERNEL(filter_instag_grad, ops::FilterInstagGradKernel<float>,
                       ops::FilterInstagGradKernel<double>,
                       ops::FilterInstagGradKernel<int32_t>,
                       ops::FilterInstagGradKernel<int64_t>);
