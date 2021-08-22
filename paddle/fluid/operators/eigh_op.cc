/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/eigh_op.h"

namespace paddle {
namespace operators {

using framework::Tensor;

class EighOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("X"), true,
                      platform::errors::InvalidArgument(
                          "Input(X) of EighOp should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasOutput("OutValue"), true,
                      platform::errors::InvalidArgument(
                          "Output(Out) of EighOp should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasOutput("OutVector"), true,
                      platform::errors::InvalidArgument(
                          "Output(Out) of EighOp should not be null."));

    auto input_dim = ctx->GetInputDim("X");

    int64_t batch_size = 1;
    for (int i = 0; i < input_dim.size() - 2; i++) {
      batch_size *= input_dim[i];
    }
    std::vector<int64_t> v_dim = {input_dim[1]};
    if (batch_size > 1) {
      v_dim = {batch_size, input_dim[1]};
    }

    PADDLE_ENFORCE_EQ(input_dim[input_dim.size() - 1],
                      input_dim[input_dim.size() - 2],
                      platform::errors::InvalidArgument(
                          "ShapeError: The input matrix must "
                          "be batches of square matrices.But received: the "
                          "'shape' of Input is [%d]",
                          input_dim.size()));

    ctx->SetOutputDim("OutValue", framework::make_ddim(v_dim));
    ctx->SetOutputDim("OutVector", input_dim);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};

class EignOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "Hermitian or real symmetric matrices whose eigenvalues and "
             "eigenvectors are to be computed ");
    AddOutput("OutValue",
              "The eigenvalues in ascending order, "
              "each repeated according to its multiplicity.");
    AddOutput(
        "OutVector",
        "The column v[:, i] is the normalized eigenvector corresponding to the,"
        "eigenvalue w[i]. Will return a matrix object if a is a matrix "
        "object.");
    AddAttr<std::string>(
        "UPLO",
        "the lower triangular part of a (‘L’, default) or the upper "
        "triangular part (‘U’)")
        .SetDefault("L");
    AddComment(R"DOC(
Eigh Operator.

Return the eigenvalues and eigenvectors of a complex Hermitian
 (conjugate symmetric) or a real symmetric matrix.

Returns two objects, a 1-D array containing the eigenvalues of a,
 and a 2-D square array or matrix (depending on the input type) 
of the corresponding eigenvectors (in columns).
)DOC");
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(eigh, ops::EighOp, ops::EignOpMaker);
REGISTER_OP_CPU_KERNEL(
    eigh, ops::EighKernel<paddle::platform::CPUDeviceContext,
                          paddle::platform::complex<float>, float>,
    ops::EighKernel<paddle::platform::CPUDeviceContext,
                    paddle::platform::complex<double>, double>,
    ops::EighKernel<paddle::platform::CPUDeviceContext, double, double>,
    ops::EighKernel<paddle::platform::CPUDeviceContext, float, float>);
