/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/sequence_ops/sequence_topk_pooling_op.h"
#include <string>

namespace paddle {
namespace operators {

class SequenceTopkPoolingOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of SequencePoolOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of SequencePoolOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("pos"),
                   "pos(out) should not be null");

    auto attr = ctx->Attrs();
    auto channel_num = attr.Get<int>("channel_num");
    auto topk = attr.Get<int>("topk");
    
    std::vector<int> vec_out_shape;
    if (ctx->IsRuntime()) {
        framework::Variable* x_var =
            boost::get<framework::Variable*>(ctx->GetInputVarPtrs("X")[0]);
        const auto& x_lod = x_var->Get<LoDTensor>().lod();
        PADDLE_ENFORCE(!x_lod.empty(), "The Input(X) must hold lod info.");
        const auto& x_lod_0 = x_lod[0];
        vec_out_shape.push_back(x_lod_0.size() - 1);
    }
    else
    {
        vec_out_shape.push_back( -1 );
    }

    vec_out_shape.push_back( channel_num * topk );

    ctx->SetOutputDim("Out", framework::make_ddim(vec_out_shape));
    ctx->ShareLoD("X", "Out");
  }
};

class SequenceTopkPoolingOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(LoDTensor) The variable-length input of SequenceTopkPoolingOp");
    AddOutput("Out",
              "(Tensor) The output of SequenceTopkPoolingOp does not contain LoD "
              "infomation.");
    AddOutput("pos",
              "(Tensor<int>) store the topk index " ).AsIntermediate();
    AddAttr<int>("topk",
              "topk attr");
    AddAttr<int>("channel_num",
              "channel number");
    AddAttr<int>("batch_size",
              "batch size");
    AddAttr<bool>("is_test",
                  "(bool, default false) Set to true for inference only, false "
                  "for training. Some layers may run faster when this is true.")
        .SetDefault(false);
    AddComment(R"DOC(
    sequecen topk pooling op
    )DOC");
  }
};

class SequenceTopkPoolingGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Gradient of Out should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("X"), "The input X should not be null.");

    ctx->ShareDim("X", /*->*/ framework::GradVarName("X"));
    ctx->ShareLoD("X", /*->*/ framework::GradVarName("X"));
  }

 protected:
 framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    //return framework::OpKernelType(
    //    framework::ToDataType(ctx.Input<Tensor>("X")->type()),
    //    ctx.device_context());
        auto data_type = framework::GetDataTypeOfVar(ctx.InputVar("X"));
        return framework::OpKernelType(data_type, ctx.device_context());
  }
};

class SequenceTopkPoolingGradOpMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    auto* op_desc_ptr = new framework::OpDesc();
    op_desc_ptr->SetType("sequence_topk_pooling_grad");
    op_desc_ptr->SetInput("X", Input("X"));
    op_desc_ptr->SetInput("pos", Output("pos"));

    op_desc_ptr->SetInput(framework::GradVarName("Out"), OutputGrad("Out"));
    op_desc_ptr->SetOutput(framework::GradVarName("X"), InputGrad("X"));
    op_desc_ptr->SetAttrMap(Attrs());
    return std::unique_ptr<framework::OpDesc>(op_desc_ptr);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(sequence_topk_pooling, ops::SequenceTopkPoolingOp, ops::SequenceTopkPoolingOpMaker,
                  paddle::framework::DefaultGradOpDescMaker<true>);
REGISTER_OPERATOR(sequence_topk_pooling_grad, ops::SequenceTopkPoolingGradOp);
REGISTER_OP_CPU_KERNEL(
    sequence_topk_pooling,
    ops::SequenceTopkPoolingKernel<float>);
REGISTER_OP_CPU_KERNEL(
    sequence_topk_pooling_grad,
    ops::SequenceTopkPoolingGradKernel<float>);
