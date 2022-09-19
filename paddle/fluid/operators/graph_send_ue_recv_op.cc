// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/multiary.h"

namespace paddle {
namespace operators {

class GraphSendUERecvOP : public framework::OperatorWithKernel {
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

class GraphSendUERecvGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    auto in_dims = ctx->GetInputDim("X");
    ctx->SetOutputDim(framework::GradVarName("X"), in_dims);
    auto y_dims = ctx->GetInputDim("Y");
    ctx->SetOutputDim(framework::GradVarName("Y"), y_dims);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.device_context());
  }
};

class GraphSendUERecvOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "The input tensor with data type float32, float64, int32, int64.");
    AddInput("Y",
             "The input edge weight tensor, data type should be same with X");
    AddInput("Src_index", "The source index tensor.");
    AddInput("Dst_index", "The destination index tensor.");
    AddInput("Out_size",
             "(Tensor<int>, optional). The 0th dimension of the output."
             "It has a higher priority than Attr(out_size).")
        .AsDispensable();
    AddOutput("Out", "Output tensor of graph_send_ue_recv op.");
    AddOutput("Dst_count",
              "Count tensor of Dst_index, mainly for MEAN reduce_op.")
        .AsIntermediate();
    AddAttr<std::string>("message_op",
                         "(string, default 'ADD')"
                         "Define differenct computation types between X and E.")
        .SetDefault("ADD")
        .InEnum({"ADD", "MUL"});
    AddAttr<std::string>("reduce_op",
                         "(string, default 'SUM')"
                         "Define different pool types to receive the result "
                         "tensors of Dst_index.")
        .SetDefault("SUM")
        .InEnum({"SUM", "MEAN", "MIN", "MAX"});
    AddAttr<std::vector<int64_t>>(
        "out_size",
        "(vector<int64_t>, default {0})"
        "Define the first dimension of Output tensor."
        "If set default {0}, then the shape of Out is the same with X.")
        .SetDefault({0});
    AddComment(R"DOC(
Graph Learning Send_UE_Recv combine operator.

$Out = Recv(Compute(Send(X, Src_index), Y, message_op), Dst_index, reduce_op)$

This operator is mainly used in Graph Learning domain, and the main purpose is to reduce
intermediate memory consumption in the process of message passing.

Take `X` as the input tensor, we first use `src_index` to gather corresponding data.
Then the gather data should compute with `Y` in different message_ops, like add, sub, mul, and div,
and get the computation result. Then, use `dst_index` to update the corresponding position of output
tensor in different pooling types, like sum, mean, max, or min.

)DOC");
  }
};

template <typename T>
class GraphSendUERecvGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("graph_send_ue_recv_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput("Y", this->Input("Y"));
    op->SetInput("Src_index", this->Input("Src_index"));
    op->SetInput("Dst_index", this->Input("Dst_index"));

    if (PADDLE_GET_CONST(std::string, this->GetAttr("reduce_op")) == "MEAN") {
      op->SetInput("Dst_count", this->Output("Dst_count"));
    }

    if (PADDLE_GET_CONST(std::string, this->GetAttr("reduce_op")) == "MIN" ||
        PADDLE_GET_CONST(std::string, this->GetAttr("reduce_op")) == "MAX") {
      op->SetInput("Out", this->Output("Out"));
    }

    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetOutput(framework::GradVarName("Y"), this->InputGrad("Y"));
    op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

DECLARE_INFER_SHAPE_FUNCTOR(graph_send_ue_recv,
                            GraphSendUERecvInferShapeFunctor,
                            PD_INFER_META(phi::GraphSendUERecvInferMeta));
REGISTER_OPERATOR(graph_send_ue_recv,
                  ops::GraphSendUERecvOP,
                  ops::GraphSendUERecvOpMaker,
                  ops::GraphSendUERecvGradOpMaker<paddle::framework::OpDesc>,
                  ops::GraphSendUERecvGradOpMaker<paddle::imperative::OpBase>,
                  GraphSendUERecvInferShapeFunctor);
REGISTER_OPERATOR(graph_send_ue_recv_grad, ops::GraphSendUERecvGradOp);
