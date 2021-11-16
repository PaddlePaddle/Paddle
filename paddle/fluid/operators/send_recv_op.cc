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

#include "paddle/fluid/operators/send_recv_op.h"

namespace paddle {
namespace operators {

class SendRecvOP : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "SendRecv");
    OP_INOUT_CHECK(ctx->HasInput("Src_index"), "Input", "Src_index",
                   "SendRecv");
    OP_INOUT_CHECK(ctx->HasInput("Dst_index"), "Input", "Dst_index",
                   "SendRecv");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "SendRecv");

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

    // TODO(daisiming): If the shape of src_index and dst_index should be same?
    auto dims = ctx->GetInputDim("X");
    ctx->SetOutputDim("Out", dims);

    if (ctx->Attrs().Get<std::string>("pool_type") == "MEAN") {
      OP_INOUT_CHECK(ctx->HasOutput("Dst_count"), "Output", "Dst_count",
                     "SendRecv");
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

class SendRecvGradOp : public framework::OperatorWithKernel {
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

class SendRecvOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "The input tensor with data type float32, "
             "float64 or float16");
    AddInput("Src_index", "The source index tensor.");
    AddInput("Dst_index", "The destination index tensor.");
    AddOutput("Out", "Output tensor of send_recv op.");
    AddOutput("Dst_count",
              "Count tensor of Dst_index, mainly for MEAN pool_type.")
        .AsIntermediate();
    AddAttr<std::string>(
        "pool_type",
        "(string, default 'SUM')"
        "Define different pool types to receive the result tensors")
        .SetDefault("SUM")
        .InEnum({"SUM", "MEAN", "MIN", "MAX"});
    // TODO(daisiming): Add a simple example here.
    AddComment(R"DOC(
SendRecv Operator.

$Out = Recv(Send(X, Src_index), Dst_index, pool_type)$

This operator 

Example:

pass
)DOC");
  }
};

template <typename T>
class SendRecvGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("send_recv_grad");
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

REGISTER_OPERATOR(send_recv, ops::SendRecvOP, ops::SendRecvOpMaker,
                  ops::SendRecvGradOpMaker<paddle::framework::OpDesc>,
                  ops::SendRecvGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(send_recv_grad, ops::SendRecvGradOp);
REGISTER_OP_CPU_KERNEL(send_recv, ops::SendRecvOpKernel<CPU, float, int>,
                       ops::SendRecvOpKernel<CPU, float, int64_t>,
                       ops::SendRecvOpKernel<CPU, double, int>,
                       ops::SendRecvOpKernel<CPU, double, int64_t>,
                       ops::SendRecvOpKernel<CPU, int, int>,
                       ops::SendRecvOpKernel<CPU, int, int64_t>,
                       ops::SendRecvOpKernel<CPU, int64_t, int>,
                       ops::SendRecvOpKernel<CPU, int64_t, int64_t>);

REGISTER_OP_CPU_KERNEL(send_recv_grad,
                       ops::SendRecvGradOpKernel<CPU, float, int>,
                       ops::SendRecvGradOpKernel<CPU, float, int64_t>,
                       ops::SendRecvGradOpKernel<CPU, double, int>,
                       ops::SendRecvGradOpKernel<CPU, double, int64_t>,
                       ops::SendRecvGradOpKernel<CPU, int, int>,
                       ops::SendRecvGradOpKernel<CPU, int, int64_t>,
                       ops::SendRecvGradOpKernel<CPU, int64_t, int>,
                       ops::SendRecvGradOpKernel<CPU, int64_t, int64_t>);
