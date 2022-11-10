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

#include <string>

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/unary.h"

namespace paddle {
namespace operators {

class ShapeOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    framework::proto::VarType::Type dafault_data_type =
        static_cast<framework::proto::VarType::Type>(-1);
    framework::proto::VarType::Type input_data_type = dafault_data_type;
    const phi::DenseTensor *t = nullptr;
    if (ctx.InputVar("Input")->IsType<phi::DenseTensor>()) {
      t = &ctx.InputVar("Input")->Get<phi::DenseTensor>();
    } else if (ctx.InputVar("Input")->IsType<phi::SelectedRows>()) {
      t = &(ctx.InputVar("Input")->Get<phi::SelectedRows>().value());
    }
    if (t != nullptr) {
      input_data_type = paddle::framework::TransToProtoVarType(t->dtype());
    }
    PADDLE_ENFORCE_NE(input_data_type,
                      dafault_data_type,
                      platform::errors::InvalidArgument(
                          "The Input Variable(Input) of (shape) Operator used "
                          "to determine kernel "
                          "data type is empty or not phi::DenseTensor or "
                          "phi::SelectedRows."));
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }

 protected:
  framework::OpKernelType GetKernelTypeForVar(
      const std::string &var_name,
      const phi::DenseTensor &tensor,
      const framework::OpKernelType &expected_kernel_type) const override {
    return framework::OpKernelType(expected_kernel_type.data_type_,
                                   expected_kernel_type.place_,
                                   tensor.layout());
  }
};

class ShapeOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Input", "(LoDTensor), The input tensor.");
    AddOutput(
        "Out",
        "(LoDTensor), The shape of input tensor, the data type of the shape"
        " is int32_t, will be on the same device with the input Tensor.");
    AddComment(R"DOC(
Shape Operator.

Return the shape of the input.
)DOC");
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(ShapeNoNeedBufferVarsInferer, "Input");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

DECLARE_INFER_SHAPE_FUNCTOR(shape,
                            ShapeInferShapeFunctor,
                            PD_INFER_META(phi::ShapeInferMeta));

REGISTER_OPERATOR(
    shape,
    ops::ShapeOp,
    ops::ShapeOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    ops::ShapeNoNeedBufferVarsInferer,
    ShapeInferShapeFunctor);
