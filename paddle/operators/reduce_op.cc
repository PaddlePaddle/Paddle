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

#include "paddle/operators/reduce_op.h"
#include "paddle/operators/net_op.h"

namespace paddle {
namespace operators {

using framework::Tensor;

class ReduceOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContextBase *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of ReduceOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of ReduceOp should not be null.");
    auto x_dims = ctx->GetInputDim("X");
    auto x_rank = x_dims.size();
    PADDLE_ENFORCE_LE(x_rank, 6, "Tensors with rank at most 6 are supported.");
    int dim = ctx->Attrs().Get<int>("dim");
    if (dim < 0) dim = x_rank + dim;
    PADDLE_ENFORCE_LT(
        dim, x_rank,
        "The dim should be in the range [-rank(input), rank(input)).");
    bool keep_dim = ctx->Attrs().Get<bool>("keep_dim");
    auto dims_vector = vectorize(x_dims);
    if (keep_dim || x_rank == 1) {
      dims_vector[dim] = 1;
    } else {
      dims_vector.erase(dims_vector.begin() + dim);
    }
    auto out_dims = framework::make_ddim(dims_vector);
    ctx->SetOutputDim("Out", out_dims);
    if (dim != 0) {
      // Only pass LoD when not reducing on the first dim.
      ctx->ShareLoD("X", /*->*/ "Out");
    }
  }
};

class ReduceGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContextBase *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should not be null.");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) should not be null.");
    auto x_dims = ctx->GetInputDim("X");
    auto x_rank = x_dims.size();
    PADDLE_ENFORCE_LE(x_rank, 6, "Tensors with rank at most 6 are supported.");
    int dim = ctx->Attrs().Get<int>("dim");
    if (dim < 0) dim = x_rank + dim;
    PADDLE_ENFORCE_LT(
        dim, x_rank,
        "The dim should be in the range [-rank(input), rank(input)).");
    auto x_grad_name = framework::GradVarName("X");
    if (ctx->HasOutput(x_grad_name)) {
      ctx->SetOutputDim(x_grad_name, x_dims);
    }
  }
};

class ReduceOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  ReduceOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput(
        "X",
        "(Tensor) The input tensor. Tensors with rank at most 6 are supported");
    AddOutput("Out", "(Tensor) The result tensor.");
    AddAttr<int>(
        "dim",
        "(int, default 1) The dimension to reduce. "
        "Must be in the range [-rank(input), rank(input)). "
        "If `dim < 0`, the dim to reduce is `rank + dim`. "
        "Noting that reducing on the first dim will make the LoD info lost.")
        .SetDefault(0);
    AddAttr<bool>("keep_dim",
                  "(bool, default false) "
                  "If true, retain the reduced dimension with length 1.")
        .SetDefault(false);
    comment_ = R"DOC(
{ReduceOP} operator computes the {reduce} of input tensor along the given dimension. 
The result tensor has 1 fewer dimension than the input unless `keep_dim` is true.
)DOC";
    AddComment(comment_);
  }

 protected:
  std::string comment_;

  void Replace(std::string &src, std::string from, std::string to) {
    std::size_t len_from = std::strlen(from.c_str());
    std::size_t len_to = std::strlen(to.c_str());
    for (std::size_t pos = src.find(from); pos != std::string::npos;
         pos = src.find(from, pos + len_to)) {
      src.replace(pos, len_from, to);
    }
  }

  void SetComment(std::string name, std::string op) {
    Replace(comment_, "{ReduceOP}", name);
    Replace(comment_, "{reduce}", op);
  }
};

class ReduceSumOpMaker : public ReduceOpMaker {
 public:
  ReduceSumOpMaker(framework::OpProto *proto,
                   framework::OpAttrChecker *op_checker)
      : ReduceOpMaker(proto, op_checker) {
    SetComment("ReduceSum", "sum");
    AddComment(comment_);
  }
};

class ReduceMeanOpMaker : public ReduceOpMaker {
 public:
  ReduceMeanOpMaker(framework::OpProto *proto,
                    framework::OpAttrChecker *op_checker)
      : ReduceOpMaker(proto, op_checker) {
    SetComment("ReduceMean", "mean");
    AddComment(comment_);
  }
};

class ReduceMaxOpMaker : public ReduceOpMaker {
 public:
  ReduceMaxOpMaker(framework::OpProto *proto,
                   framework::OpAttrChecker *op_checker)
      : ReduceOpMaker(proto, op_checker) {
    SetComment("ReduceMax", "max");
    AddComment(comment_);
  }
};

class ReduceMinOpMaker : public ReduceOpMaker {
 public:
  ReduceMinOpMaker(framework::OpProto *proto,
                   framework::OpAttrChecker *op_checker)
      : ReduceOpMaker(proto, op_checker) {
    SetComment("ReduceMin", "min");
    AddComment(comment_);
  }
};

class NormOp : public NetOp {
 public:
  NormOp(const std::string &type, const framework::VariableNameMap &inputs,
         const framework::VariableNameMap &outputs,
         const framework::AttributeMap &attrs)
      : NetOp(type, inputs, outputs, attrs) {
    PADDLE_ENFORCE_NE(Input("X"), framework::kEmptyVarName,
                      "Input(X) of NormOp should not be null.");
    PADDLE_ENFORCE_NE(Output("AbsOut"), framework::kEmptyVarName,
                      "Output(AbsOut) of NormOp should not be null.");
    PADDLE_ENFORCE_NE(Output("PowOut"), framework::kEmptyVarName,
                      "Output(PowOut) of NormOp should not be null.");
    PADDLE_ENFORCE_NE(Output("SumOut"), framework::kEmptyVarName,
                      "Output(SumOut) of NormOp should not be null.");
    PADDLE_ENFORCE_NE(Output("Out"), framework::kEmptyVarName,
                      "Output(Out) of NormOp should not be null.");
    auto dim = Attr<int>("dim");
    auto keep_dim = Attr<bool>("keep_dim");
    auto p = Attr<float>("p");
    PADDLE_ENFORCE_GT(p, 0, "Order of the norm should be positive.");
    AppendOp(framework::OpRegistry::CreateOp("abs", {{"X", {Input("X")}}},
                                             {{"Y", {Output("AbsOut")}}}, {}));
    AppendOp(framework::OpRegistry::CreateOp("pow", {{"X", {Output("AbsOut")}}},
                                             {{"Y", {Output("PowOut")}}},
                                             {{"factor", p}}));
    framework::AttributeMap sum_attr;
    sum_attr["dim"] = dim;
    sum_attr["keep_dim"] = keep_dim;
    AppendOp(framework::OpRegistry::CreateOp(
        "reduce_sum", {{"X", {Output("PowOut")}}},
        {{"Out", {Output("SumOut")}}}, sum_attr));
    AppendOp(framework::OpRegistry::CreateOp(
        "pow", {{"X", {Output("SumOut")}}}, {{"Y", {Output("Out")}}},
        {{"factor", static_cast<float>(1. / p)}}));
    CompleteAddOp(false);
  }
};

class NormOpMaker : public ReduceOpMaker {
 public:
  NormOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : ReduceOpMaker(proto, op_checker) {
    AddOutput("AbsOut",
              "(Tensor) The intermediate output of Norm operator, "
              "saving the absolute value of the input tensor X.")
        .AsIntermediate();
    AddOutput("PowOut",
              "(Tensor) The intermediate output of Norm operator, "
              "saving the p-th power of the output tensor AbsOut.")
        .AsIntermediate();
    AddOutput("SumOut",
              "(Tensor) the intermediate output of Norm operator, "
              "saving the sum of PowOut reduced on the given dimension.")
        .AsIntermediate();
    AddAttr<float>("p", "(float, default 2) The order of Norm.").SetDefault(2);
    SetComment("Norm", "vector p-norm");
    AddComment(comment_);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP(reduce_sum, ops::ReduceOp, ops::ReduceSumOpMaker, reduce_sum_grad,
            ops::ReduceGradOp);
REGISTER_OP_CPU_KERNEL(
    reduce_sum,
    ops::ReduceKernel<paddle::platform::CPUPlace, float, ops::SumFunctor>);
REGISTER_OP_CPU_KERNEL(reduce_sum_grad,
                       ops::ReduceGradKernel<paddle::platform::CPUPlace, float,
                                             ops::SumGradFunctor>);

REGISTER_OP(reduce_mean, ops::ReduceOp, ops::ReduceMeanOpMaker,
            reduce_mean_grad, ops::ReduceGradOp);
REGISTER_OP_CPU_KERNEL(
    reduce_mean,
    ops::ReduceKernel<paddle::platform::CPUPlace, float, ops::MeanFunctor>);
REGISTER_OP_CPU_KERNEL(reduce_mean_grad,
                       ops::ReduceGradKernel<paddle::platform::CPUPlace, float,
                                             ops::MeanGradFunctor>);

REGISTER_OP(reduce_max, ops::ReduceOp, ops::ReduceMaxOpMaker, reduce_max_grad,
            ops::ReduceGradOp);
REGISTER_OP_CPU_KERNEL(
    reduce_max,
    ops::ReduceKernel<paddle::platform::CPUPlace, float, ops::MaxFunctor>);
REGISTER_OP_CPU_KERNEL(reduce_max_grad,
                       ops::ReduceGradKernel<paddle::platform::CPUPlace, float,
                                             ops::MaxOrMinGradFunctor>);

REGISTER_OP(reduce_min, ops::ReduceOp, ops::ReduceMaxOpMaker, reduce_min_grad,
            ops::ReduceGradOp);
REGISTER_OP_CPU_KERNEL(
    reduce_min,
    ops::ReduceKernel<paddle::platform::CPUPlace, float, ops::MinFunctor>);
REGISTER_OP_CPU_KERNEL(reduce_min_grad,
                       ops::ReduceGradKernel<paddle::platform::CPUPlace, float,
                                             ops::MaxOrMinGradFunctor>);

REGISTER_OP_WITHOUT_GRADIENT(norm, ops::NormOp, ops::NormOpMaker);
