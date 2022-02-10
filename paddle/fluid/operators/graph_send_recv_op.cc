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

#include "paddle/fluid/operators/graph_send_recv_op.h"

namespace paddle {
namespace operators {

class GraphSendRecvOP : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "GraphSendRecv");
    OP_INOUT_CHECK(ctx->HasInput("Src_index"), "Input", "Src_index",
                   "GraphSendRecv");
    OP_INOUT_CHECK(ctx->HasInput("Dst_index"), "Input", "Dst_index",
                   "GraphSendRecv");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "GraphSendRecv");

    auto src_index_dims = ctx->GetInputDim("Src_index");
    if (src_index_dims.size() == 2) {
      PADDLE_ENFORCE_EQ(src_index_dims[1], 1,
                        platform::errors::InvalidArgument(
                            "The last dim of Src_index should be 1 when it "
                            "is 2D, but we get %d",
                            src_index_dims[1]));
    } else {
      PADDLE_ENFORCE_EQ(
          src_index_dims.size(), 1,
          platform::errors::InvalidArgument(
              "The Src_index should be 1D, when it is not 2D, but we get %d",
              src_index_dims.size()));
    }

    auto dst_index_dims = ctx->GetInputDim("Dst_index");
    if (dst_index_dims.size() == 2) {
      PADDLE_ENFORCE_EQ(dst_index_dims[1], 1,
                        platform::errors::InvalidArgument(
                            "The last dim of Dst_index should be 1 when it "
                            "is 2D, but we get %d",
                            dst_index_dims[1]));
    } else {
      PADDLE_ENFORCE_EQ(
          dst_index_dims.size(), 1,
          platform::errors::InvalidArgument("The Dst_index should be 1D, "
                                            "when it is not 2D, but we get %d",
                                            dst_index_dims.size()));
    }

    PADDLE_ENFORCE_EQ(
        src_index_dims[0], dst_index_dims[0],
        platform::errors::InvalidArgument(
            "Src_index and Dst_index should have the same shape."));

    auto dims = ctx->GetInputDim("X");
    ctx->SetOutputDim("Out", dims);

    if (ctx->Attrs().Get<std::string>("pool_type") == "MEAN") {
      OP_INOUT_CHECK(ctx->HasOutput("Dst_count"), "Output", "Dst_count",
                     "GraphSendRecv");
      ctx->SetOutputDim("Dst_count", {dims[0]});
    }
  }

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
    auto in_dims = ctx->GetInputDim(framework::GradVarName("Out"));
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

    if (BOOST_GET_CONST(std::string, this->GetAttr("pool_type")) == "MEAN") {
      op->SetInput("Dst_count", this->Output("Dst_count"));
    }

    if (BOOST_GET_CONST(std::string, this->GetAttr("pool_type")) == "MIN" ||
        BOOST_GET_CONST(std::string, this->GetAttr("pool_type")) == "MAX") {
      op->SetInput("X", this->Input("X"));
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
using CPU = paddle::platform::CPUDeviceContext;

REGISTER_OPERATOR(graph_send_recv, ops::GraphSendRecvOP,
                  ops::GraphSendRecvOpMaker,
                  ops::GraphSendRecvGradOpMaker<paddle::framework::OpDesc>,
                  ops::GraphSendRecvGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(graph_send_recv_grad, ops::GraphSendRecvGradOp);
REGISTER_OP_CPU_KERNEL(graph_send_recv, ops::GraphSendRecvOpKernel<CPU, float>,
                       ops::GraphSendRecvOpKernel<CPU, double>,
                       ops::GraphSendRecvOpKernel<CPU, int>,
                       ops::GraphSendRecvOpKernel<CPU, int64_t>);

REGISTER_OP_CPU_KERNEL(graph_send_recv_grad,
                       ops::GraphSendRecvGradOpKernel<CPU, float>,
                       ops::GraphSendRecvGradOpKernel<CPU, double>,
                       ops::GraphSendRecvGradOpKernel<CPU, int>,
                       ops::GraphSendRecvGradOpKernel<CPU, int64_t>);
