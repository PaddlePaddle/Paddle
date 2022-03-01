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

#include "paddle/fluid/operators/eigvals_op.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {
class EigvalsOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(Tensor), A complex- or real-valued tensor with shape (*, n, n)"
             "where * is zero or more batch dimensions");
    AddOutput("Out",
              "(Tensor) The output tensor with shape (*,n) cointaining the "
              "eigenvalues of X.");
    AddComment(R"DOC(eigvals operator
        Return the eigenvalues of one or more square matrices. The eigenvalues are complex even when the input matrices are real.
        )DOC");
  }
};

class EigvalsOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "Eigvals");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "Eigvals");

    DDim x_dims = ctx->GetInputDim("X");
    PADDLE_ENFORCE_GE(x_dims.size(), 2,
                      platform::errors::InvalidArgument(
                          "The dimensions of Input(X) for Eigvals operator "
                          "should be at least 2, "
                          "but received X's dimension = %d, X's shape = [%s].",
                          x_dims.size(), x_dims));

    if (ctx->IsRuntime() || !phi::contain_unknown_dim(x_dims)) {
      int last_dim = x_dims.size() - 1;
      PADDLE_ENFORCE_EQ(x_dims[last_dim], x_dims[last_dim - 1],
                        platform::errors::InvalidArgument(
                            "The last two dimensions of Input(X) for Eigvals "
                            "operator should be equal, "
                            "but received X's shape = [%s].",
                            x_dims));
    }

    auto output_dims = vectorize(x_dims);
    output_dims.resize(x_dims.size() - 1);
    ctx->SetOutputDim("Out", phi::make_ddim(output_dims));
  }
};

class EigvalsOpVarTypeInference : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext* ctx) const {
    auto input_dtype = ctx->GetInputDataType("X");
    auto output_dtype = framework::IsComplexType(input_dtype)
                            ? input_dtype
                            : framework::ToComplexType(input_dtype);
    ctx->SetOutputDataType("Out", output_dtype);
  }
};
}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OPERATOR(eigvals, ops::EigvalsOp, ops::EigvalsOpMaker,
                  ops::EigvalsOpVarTypeInference);
REGISTER_OP_CPU_KERNEL(eigvals,
                       ops::EigvalsKernel<plat::CPUDeviceContext, float>,
                       ops::EigvalsKernel<plat::CPUDeviceContext, double>,
                       ops::EigvalsKernel<plat::CPUDeviceContext,
                                          paddle::platform::complex<float>>,
                       ops::EigvalsKernel<plat::CPUDeviceContext,
                                          paddle::platform::complex<double>>);
