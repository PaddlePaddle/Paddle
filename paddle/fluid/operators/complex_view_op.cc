// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/complex_view_op.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace operators {

class AsComplexOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "as_complex");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "as_complex");

    auto in_dims = ctx->GetInputDim("X");
    const int input_rank = in_dims.size();
    PADDLE_ENFORCE_GE(
        input_rank, 1,
        platform::errors::InvalidArgument(
            "The rank of input(X) is less than 1. "
            "Expected the rank of input(X) to be equal to or greater than 1."
            "But received rank of input(X) = %d",
            input_rank));
    const int last_dim_size = in_dims[input_rank - 1];
    PADDLE_ENFORCE_EQ(
        last_dim_size, 2,
        platform::errors::InvalidArgument(
            "The size of the last dimension of input(X)"
            "does not equals 2."
            "Expected the size of last dimension of input(X) to be 2."
            "But received %d",
            last_dim_size));

    const framework::DDim out_dims(in_dims.Get(), input_rank - 1);
    ctx->SetOutputDim("Out", out_dims);
    ctx->ShareLoD("X", /*->*/ "Out");
  }
};

class AsComplexOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), The input tensor of view_as_complex op.");
    AddOutput("Out", "(Tensor), The output tensor of view_as_complex op.");
    AddComment(R"DOC(
As_complex Operator.

This operator is used to return a complex tensor represented
by an old-fashioned real tensor. The size of the last dimension of 
the input tensor should be 2, which corresponds to 'real' and 
'complex', respectively.

)DOC");
  }
};

template <typename T>
class AsComplexGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  void Apply(GradOpPtr<T> retv) const override {
    retv->SetType("as_real");
    retv->SetInput("X", this->OutputGrad("Out"));
    retv->SetAttrMap(this->Attrs());
    retv->SetOutput("Out", this->InputGrad("X"));
  }
};

class AsRealOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "as_real");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "as_real");

    auto out_dims_v = framework::vectorize(ctx->GetInputDim("X"));
    out_dims_v.push_back(2);
    const framework::DDim out_dims = framework::make_ddim(out_dims_v);
    ctx->SetOutputDim("Out", out_dims);
    ctx->ShareLoD("X", /*->*/ "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto input_data_type =
        framework::OperatorWithKernel::IndicateVarDataType(ctx, "X");
    return framework::OpKernelType(framework::ToRealType(input_data_type),
                                   ctx.GetPlace());
  }
};

class AsRealOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), The input tensor of as_real op.");
    AddOutput("Out", "(Tensor), The output tensor of as_real op.");
    AddComment(R"DOC(
AsReal Operator.

This operator is used to return an old-fashioned real tensor from a 
complex tensor. The size of the last dimension of the output tensor is 2,
which corresponds to 'real' and 'complex', respectively.

)DOC");
  }
};

template <typename T>
class AsRealGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  void Apply(GradOpPtr<T> retv) const override {
    retv->SetType("as_complex");
    retv->SetInput("X", this->OutputGrad("Out"));
    retv->SetAttrMap(this->Attrs());
    retv->SetOutput("Out", this->InputGrad("X"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(as_complex, ops::AsComplexOp, ops::AsComplexOpMaker,
                  ops::AsComplexGradMaker<paddle::framework::OpDesc>,
                  ops::AsComplexGradMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(as_real, ops::AsRealOp, ops::AsRealOpMaker,
                  ops::AsRealGradMaker<paddle::framework::OpDesc>,
                  ops::AsRealGradMaker<paddle::imperative::OpBase>);

REGISTER_OP_CPU_KERNEL(
    as_complex, ops::AsComplexKernel<paddle::platform::CPUDeviceContext, float>,
    ops::AsComplexKernel<paddle::platform::CPUDeviceContext, double>);

REGISTER_OP_CPU_KERNEL(
    as_real, ops::AsRealKernel<paddle::platform::CPUDeviceContext, float>,
    ops::AsRealKernel<paddle::platform::CPUDeviceContext, double>);
