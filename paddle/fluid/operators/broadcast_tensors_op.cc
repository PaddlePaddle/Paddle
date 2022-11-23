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

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/var_type_inference.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/multiary.h"

namespace paddle {
namespace operators {
using framework::DDim;

class BroadcastTensorsOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    // Broadcast semantics enforces all input variables having the same
    // DataType/VarType
    // This condition is also checked during VarType Inference
    // Here we simply copy input type to output
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
  }
};

class BroadcastTensorsOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "A Varaible list. The shape and data type of the list elements"
             "should be consistent. Variable can be multi-dimensional Tensor"
             "or LoDTensor, and data types can be: bool, float16, float32, "
             "float64, int32, "
             "int64.")
        .AsDuplicable();
    AddOutput("Out",
              "the sum of input :code:`x`. its shape and data types are "
              "consistent with :code:`x`.")
        .AsDuplicable();
    AddComment(
        R"DOC(This OP is used to broadcast a vector of inputs
                     with Tensor or LoDTensor type, following broadcast semantics.)DOC");
  }
};

class BroadcastTensorsOpVarTypeInference : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext* ctx) const override {
    // We need at least two tensors to satisfy broadcast semantics
    size_t input_size = ctx->InputSize("X");
    PADDLE_ENFORCE_GT(
        input_size,
        0,
        platform::errors::InvalidArgument(
            "BroadcastTensorsOp should have at least one input variables,"
            "but only received %d ",
            input_size));

    // BroadcastTensorsOp takes a vector of variables named "X"
    // Here we loop through input variables,
    // and check if their DataType/VarType are the same
    auto var_type = ctx->GetInputType("X", 0);
    auto data_type = ctx->GetInputDataType("X", 0);
    for (size_t ind = 1; ind < input_size; ind++) {
      auto cur_var_type = ctx->GetInputType("X", ind);
      PADDLE_ENFORCE_EQ(
          var_type,
          cur_var_type,
          platform::errors::InvalidArgument(
              "inputs to BroadcastTensorsOp should have the same variable type,"
              "but detected %d v.s %d ",
              framework::ToTypeName(var_type),
              framework::ToTypeName(cur_var_type)));

      auto cur_data_type = ctx->GetInputDataType("X", ind);
      PADDLE_ENFORCE_EQ(
          data_type,
          cur_data_type,
          platform::errors::InvalidArgument(
              "inputs to BroadcastTensorsOp should have the same data type,"
              "but detected %d v.s %d ",
              framework::ToTypeName(var_type),
              framework::ToTypeName(cur_var_type)));
    }

    // Outputs having the same DataType/VarType as inputs
    ctx->SetOutputType("Out", var_type, framework::ALL_ELEMENTS);
    ctx->SetOutputDataType("Out", data_type, framework::ALL_ELEMENTS);
  }
};

/* ------ BroadcastTensorsGradOp ------ */
class BroadcastTensorsGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasOutputs(framework::GradVarName("X")),
                   "Output",
                   "X@grad",
                   "broadcast_tensors");
    OP_INOUT_CHECK(ctx->HasInputs("X"), "Input", "X", "broadcast_tensors");
    OP_INOUT_CHECK(ctx->HasInputs(framework::GradVarName("Out")),
                   "Input",
                   "Out@grad",
                   "broadcast_tensors");

    const auto& forward_input_dims = ctx->GetInputsDim("X");
    ctx->SetOutputsDim(framework::GradVarName("X"), forward_input_dims);
    ctx->ShareAllLoD("X", /*->*/ framework::GradVarName("X"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.device_context());
  }
};

template <typename T>
class BroadcastTensorsGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType("broadcast_tensors_grad");
    // We need "X" only for backward shape inference
    grad_op->SetInput("X", this->Input("X"));
    grad_op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    grad_op->SetOutput(framework::GradVarName("X"),
                       this->InputGrad("X", /* drop_empty_grad */ false));
    grad_op->SetAttrMap(this->Attrs());
  }
};

class BroadcastTensorsGradOpVarTypeInference
    : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext* ctx) const override {
    auto var_type = ctx->GetInputType("X", 0);
    auto data_type = ctx->GetInputDataType("X", 0);

    ctx->SetOutputType(
        framework::GradVarName("X"), var_type, framework::ALL_ELEMENTS);
    ctx->SetOutputDataType(
        framework::GradVarName("X"), data_type, framework::ALL_ELEMENTS);
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(BroadcastTensorsGradNoNeedBufVarsInferer,
                                    "X");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

DECLARE_INFER_SHAPE_FUNCTOR(broadcast_tensors,
                            BroadcastTensorsInferShapeFunctor,
                            PD_INFER_META(phi::BroadcastTensorsInferMeta));

REGISTER_OPERATOR(broadcast_tensors,
                  ops::BroadcastTensorsOp,
                  ops::BroadcastTensorsOpMaker,
                  ops::BroadcastTensorsGradOpMaker<paddle::framework::OpDesc>,
                  ops::BroadcastTensorsGradOpMaker<paddle::imperative::OpBase>,
                  ops::BroadcastTensorsOpVarTypeInference,
                  BroadcastTensorsInferShapeFunctor);

REGISTER_OPERATOR(broadcast_tensors_grad,
                  ops::BroadcastTensorsGradOp,
                  ops::BroadcastTensorsGradOpVarTypeInference,
                  ops::BroadcastTensorsGradNoNeedBufVarsInferer);
