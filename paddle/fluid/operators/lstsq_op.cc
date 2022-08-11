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

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/phi/infermeta/binary.h"

namespace paddle {
namespace operators {

class LstsqOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

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
    AddOutput("Residuals",
              "(Tensor), The output Residuals tensor with shape (*, k).")
        .AsDispensable();
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

DECLARE_INFER_SHAPE_FUNCTOR(lstsq,
                            LstsqInferShapeFunctor,
                            PD_INFER_META(phi::LstsqInferMeta));

REGISTER_OPERATOR(lstsq,
                  ops::LstsqOp,
                  ops::LstsqOpMaker,
                  LstsqInferShapeFunctor);

REGISTER_OP_VERSION(lstsq).AddCheckpoint(
    R"ROC(
        Upgrade lstsq, add 1 outputs [Residuals].
      )ROC",
    paddle::framework::compatible::OpVersionDesc().NewOutput(
        "Residuals",
        "Output tensor of lstsq operator, "
        "meaning the squared residuals of the calculated solutions."));
