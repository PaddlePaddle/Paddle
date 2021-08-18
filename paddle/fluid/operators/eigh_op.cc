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
// #include <memory>
// #include <string>
// #include <vector>

namespace paddle {
namespace operators {

using framework::Tensor;
// using complex64 = paddle::platform::complex<float>;
// using complex128 = paddle::platform::complex<double>;

class EighOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("X"), true,
                      platform::errors::InvalidArgument(
                          "Input(X) of EighOp should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasOutput("OutVector"), true,
                      platform::errors::InvalidArgument(
                          "Output(Out) of EighOp should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasOutput("OutValue"), true,
                      platform::errors::InvalidArgument(
                          "Output(Out) of EighOp should not be null."));

    auto input_dim = ctx->GetInputDim("X");
    int batch = 1;
    if (input_dim.size() == 3) {
      batch = input_dim[0];
    }
    std::vector<int64_t> v_dim = {batch, input_dim[1]};

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
    AddOutput("OutVector",
              "The eigenvalues in ascending order, "
              "each repeated according to its multiplicity.");
    AddOutput(
        "OutValue",
        "The column v[:, i] is the normalized eigenvector corresponding to the,"
        "eigenvalue w[i]. Will return a matrix object if a is a matrix "
        "object.");
    AddAttr<bool>("UPLO",
                  "the lower triangular part of a (‘L’, default) or the upper "
                  "triangular part (‘U’)")
        .SetDefault(true);
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
