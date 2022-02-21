// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/lstsq_op.h"
#include <string>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

class LstsqOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "LstsqOp");
    OP_INOUT_CHECK(ctx->HasInput("Y"), "Input", "Y", "LstsqOp");

    OP_INOUT_CHECK(ctx->HasOutput("Solution"), "Output", "Solution", "LstsqOp");
    OP_INOUT_CHECK(ctx->HasOutput("Rank"), "Output", "Rank", "LstsqOp");
    OP_INOUT_CHECK(ctx->HasOutput("SingularValues"), "Output", "SingularValues",
                   "LstsqOp");

    auto x_dims = ctx->GetInputDim("X");
    auto y_dims = ctx->GetInputDim("Y");
    int x_rank = x_dims.size();
    int y_rank = y_dims.size();

    PADDLE_ENFORCE_GE(x_rank, 2,
                      platform::errors::InvalidArgument(
                          "Expects input tensor x to be not less than "
                          "2 dimentions, but got dimention %d",
                          x_rank));
    PADDLE_ENFORCE_GE(y_rank, 2,
                      platform::errors::InvalidArgument(
                          "Expects input tensor y to be not less than "
                          "2 dimentions, but got dimention %d",
                          y_rank));

    PADDLE_ENFORCE_EQ(
        x_rank, y_rank,
        platform::errors::InvalidArgument(
            "Expects input tensor x and y to have the same dimension "
            "but got x's dimention [%d] and y's dimention [%d]",
            x_rank, y_rank));

    std::vector<int> batch_dims_vec{};
    for (int i = 0; i < x_rank - 2; ++i) {
      PADDLE_ENFORCE_EQ(
          x_dims[i], y_dims[i],
          platform::errors::InvalidArgument(
              "Expects input tensor x and y to have the same batch "
              "dimension, but got x's batch dimention [%d] and "
              "y's batch dimention [%d] in %d-th dim",
              x_dims[i], y_dims[i], i));
      batch_dims_vec.emplace_back(x_dims[i]);
    }

    PADDLE_ENFORCE_EQ(
        x_dims[x_rank - 2], y_dims[y_rank - 2],
        platform::errors::InvalidArgument(
            "Expects input tensor x and y to have the same row dimension "
            "of the inner-most 2-dims matrix, "
            "but got x's row dimention [%d] and y's row dimention [%d]",
            x_dims[x_rank - 2], y_dims[y_rank - 2]));

    ctx->SetOutputDim("Rank", phi::make_ddim(batch_dims_vec));

    batch_dims_vec.emplace_back(
        std::min(x_dims[x_rank - 2], x_dims[x_rank - 1]));
    ctx->SetOutputDim("SingularValues", phi::make_ddim(batch_dims_vec));

    batch_dims_vec[x_rank - 2] = x_dims[x_rank - 1];
    batch_dims_vec.emplace_back(y_dims[x_rank - 1]);
    ctx->SetOutputDim("Solution", phi::make_ddim(batch_dims_vec));
  }

 protected:
  // The output of lstsq is always complex-valued even for real-valued inputs
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto dtype = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    if (dtype != framework::proto::VarType::FP32 &&
        dtype != framework::proto::VarType::FP64) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "unsupported data type: %s!", dtype));
    }
    return framework::OpKernelType(dtype, ctx.GetPlace());
  }
};

class LstsqOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(Tensor), A real-valued tensor with shape (*, m, n). "
             "The accepted datatype is one of float32, float64");
    AddInput("Y",
             "(Tensor), A real-valued tensor with shape (*, m, k). "
             "The accepted datatype is one of float32, float64");
    AddAttr<float>(
        "rcond",
        "(float, default 0.0), A float value used to determine the effective "
        "rank of A.")
        .SetDefault(0.0f);
    AddAttr<std::string>("driver",
                         "(string, default \"gels\"). "
                         "name of the LAPACK method to be used.")
        .SetDefault("gels");
    AddOutput("Solution",
              "(Tensor), The output Solution tensor with shape (*, n, k).");
    AddOutput("Rank", "(Tensor), The output Rank tensor with shape (*).");
    AddOutput(
        "SingularValues",
        "(Tensor), The output SingularValues tensor with shape (*, min(m,n)).");
    AddComment(R"DOC(
        Lstsq Operator.
This API processes Lstsq functor for general matrices.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(lstsq, ops::LstsqOp, ops::LstsqOpMaker)

REGISTER_OP_CPU_KERNEL(
    lstsq, ops::LstsqCPUKernel<paddle::platform::CPUDeviceContext, float>,
    ops::LstsqCPUKernel<paddle::platform::CPUDeviceContext, double>);