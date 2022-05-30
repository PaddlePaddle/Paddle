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
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/ternary.h"

namespace paddle {
namespace operators {

class GraphSendRecvOP : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        ctx.device_context());
  }
};

class GraphSendRecvGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    auto in_dims = ctx->GetInputDim("X");
    ctx->SetOutputDim(framework::GradVarName("X"), in_dims);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.device_context());
  }
};

class GraphSendRecvOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "The input tensor with data type float32, float64, int32, int64.");
    AddInput("Src_index", "The source index tensor.");
    AddInput("Dst_index", "The destination index tensor.");
    AddOutput("Out", "Output tensor of graph_send_recv op.");
    AddOutput("Dst_count",
              "Count tensor of Dst_index, mainly for MEAN pool_type.")
        .AsIntermediate();
    AddAttr<std::string>("pool_type",
                         "(string, default 'SUM')"
                         "Define different pool types to receive the result "
                         "tensors of Dst_index.")
        .SetDefault("SUM")
        .InEnum({"SUM", "MEAN", "MIN", "MAX"});
    AddAttr<int64_t>(
        "out_size",
        "(int64_t, default 0)"
        "Define the first dimension of Output tensor."
        "If set default 0, then the shape of Out is the same with X.")
        .SetDefault(0);
    AddComment(R"DOC(
Graph Learning Send_Recv combine operator.

$Out = Recv(Send(X, Src_index), Dst_index, pool_type)$

This operator is mainly used in Graph Learning domain, and the main purpose is to reduce 
intermediate memory consumption in the process of message passing. 
Take `x` as the input tensor, we first use `src_index` to gather corresponding data, 
and then use `dst_index` to update the corresponding position of output tensor in different 
pooling types, like sum, mean, max, or min.

)DOC");
  }
};

template <typename T>
class GraphSendRecvGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("graph_send_recv_grad");
    op->SetInput("Src_index", this->Input("Src_index"));
    op->SetInput("Dst_index", this->Input("Dst_index"));
    op->SetInput("X", this->Input("X"));

    if (BOOST_GET_CONST(std::string, this->GetAttr("pool_type")) == "MEAN") {
      op->SetInput("Dst_count", this->Output("Dst_count"));
    }

    if (BOOST_GET_CONST(std::string, this->GetAttr("pool_type")) == "MIN" ||
        BOOST_GET_CONST(std::string, this->GetAttr("pool_type")) == "MAX") {
      op->SetInput("Out", this->Output("Out"));
    }

    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

DECLARE_INFER_SHAPE_FUNCTOR(graph_send_recv, GraphSendRecvInferShapeFunctor,
                            PD_INFER_META(phi::GraphSendRecvInferMeta));
REGISTER_OPERATOR(graph_send_recv, ops::GraphSendRecvOP,
                  ops::GraphSendRecvOpMaker,
                  ops::GraphSendRecvGradOpMaker<paddle::framework::OpDesc>,
                  ops::GraphSendRecvGradOpMaker<paddle::imperative::OpBase>,
                  GraphSendRecvInferShapeFunctor);
REGISTER_OPERATOR(graph_send_recv_grad, ops::GraphSendRecvGradOp);
