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

#include <string>

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/ternary.h"

namespace paddle {
namespace operators {

class LinspaceOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        framework::proto::VarType::Type(ctx.Attr<int>("dtype")),
        ctx.GetPlace());
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string &var_name,
      const phi::DenseTensor &tensor,
      const framework::OpKernelType &expected_kernel_type) const override {
    if (platform::is_xpu_place(tensor.place())) {
      return framework::OpKernelType(
          expected_kernel_type.data_type_, tensor.place(), tensor.layout());
    }
    return expected_kernel_type;
  }
};

class LinspaceOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Start",
             "First entry in the sequence. It is a tensor of shape [1], should "
             "be of type float32 or float64.");
    AddInput("Stop",
             "Last entry in the sequence. It is a tensor of shape [1], should "
             "be of type float32 or float64.");
    AddInput("Num",
             "Number of entry in the sequence. It is a tensor of shape [1], "
             "should be of type int32.");
    AddAttr<int>("dtype", "The output data type.");
    AddOutput("Out", "A sequence of numbers.");
    AddComment(R"DOC(
    Return fixed number of evenly spaced values within a given interval. First entry is start, and last entry is stop. In the case when Num is 1, only Start is returned. Like linspace function of numpy.
)DOC");
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
DECLARE_INFER_SHAPE_FUNCTOR(linspace,
                            LinspaceInferShapeFunctor,
                            PD_INFER_META(phi::LinspaceRawInferMeta));
REGISTER_OPERATOR(
    linspace,
    ops::LinspaceOp,
    ops::LinspaceOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    LinspaceInferShapeFunctor);

REGISTER_OP_VERSION(linspace).AddCheckpoint(
    R"ROC(
      Upgrade linspace to add a new attribute [dtype].
    )ROC",
    paddle::framework::compatible::OpVersionDesc().NewAttr(
        "dtype", "In order to change output data type ", 5));
