/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/unary.h"

namespace paddle {
namespace operators {

class SizeOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto dtype = framework::proto::VarType::FP32;  // dtype is not important
    return framework::OpKernelType(dtype, ctx.GetPlace());
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string& var_name,
      const phi::DenseTensor& tensor,
      const framework::OpKernelType& expected_kernel_type) const override {
    return expected_kernel_type;
  }
};

class SizeOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Input", "The input tensor.");
    AddOutput("Out",
              "The returned tensor, the data type "
              "is int64_t, will be on the same device with the input Tensor.");
    AddComment(R"DOC(
Size Operator.

Return the number of elements in the input.
)DOC");
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(SizeOpNoNeedBufferVarInferer, "Input");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
DECLARE_INFER_SHAPE_FUNCTOR(size,
                            SizeInferShapeFunctor,
                            PD_INFER_META(phi::SizeInferMeta));
REGISTER_OPERATOR(
    size,
    ops::SizeOp,
    ops::SizeOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    SizeInferShapeFunctor,
    ops::SizeOpNoNeedBufferVarInferer);
