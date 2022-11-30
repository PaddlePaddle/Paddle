/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/phi/infermeta/nullary.h"

namespace paddle {
namespace operators {

class EmptyOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("ShapeTensor",
             "(Tensor<int>), optional). The shape of the output."
             "It has a higher priority than Attr(shape).")
        .AsDispensable();
    AddInput("ShapeTensorList",
             "(vector<Tensor<int>>, optional). The shape of the output. "
             "It has a higher priority than Attr(shape)."
             "The shape of the element in vector must be [1].")
        .AsDuplicable()
        .AsDispensable();
    AddAttr<std::vector<int64_t>>("shape",
                                  "(vector<int64_t>) The shape of the output")
        .SetDefault({});
    AddAttr<int>("dtype", "The data type of output tensor, Default is float")
        .SetDefault(framework::proto::VarType::FP32);
    AddOutput("Out", "(Tensor) The output tensor.");
    AddComment(R"DOC(empty operator
Returns a tensor filled with uninitialized data. The shape of the tensor is
defined by the variable argument shape.


The type of the tensor is specify by `dtype`.
)DOC");
  }
};

class EmptyOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetKernelTypeForVar(
      const std::string& var_name,
      const phi::DenseTensor& tensor,
      const framework::OpKernelType& expected_kernel_type) const override {
    if (var_name == "ShapeTensor" || var_name == "ShapeTensorList") {
      return expected_kernel_type;
    } else {
      return framework::OpKernelType(
          expected_kernel_type.data_type_, tensor.place(), tensor.layout());
    }
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& context) const override {
    return framework::OpKernelType(
        framework::proto::VarType::Type(context.Attr<int>("dtype")),
        context.GetPlace());
  }
};

class EmptyOpVarTypeInference : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext* context) const override {
    auto data_type = static_cast<framework::proto::VarType::Type>(
        PADDLE_GET_CONST(int, context->GetAttr("dtype")));
    context->SetOutputDataType("Out", data_type);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

DECLARE_INFER_SHAPE_FUNCTOR(empty,
                            EmptyInferShapeFunctor,
                            PD_INFER_META(phi::CreateInferMeta));
REGISTER_OP_WITHOUT_GRADIENT(empty,
                             ops::EmptyOp,
                             ops::EmptyOpMaker,
                             ops::EmptyOpVarTypeInference,
                             EmptyInferShapeFunctor);
