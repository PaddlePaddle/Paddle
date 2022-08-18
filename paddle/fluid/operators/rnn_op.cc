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

#include <memory>
#include <string>

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/multiary.h"

namespace paddle {
namespace operators {

class RNNOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "Input"),
        ctx.device_context());
  }
};

class RNNOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput(
        "Input",
        "(Tensor) RNN input tensor, which support variable-time length input "
        "sequence."
        "The shape of the Tensor MUST be ( seq_len * batch_size * input_size)"
        "seq_len is the total time step in this mini-batch (CAN be change in "
        "different batch)"
        "batch_size is the instance number of this batch"
        "input_size is the hidden size of the input."
        "input_size and the hidden_size in the next may not be same");
    AddInput("PreState",
             "(Tensor) the initial hidden state of the LSTM"
             "input. This is a tensor with shape (num_layers x batch_size x "
             "hidden_size)"
             "and When is_bidirec is True, the shape will be (num_layers*2 x "
             "batch_size x hidden_size)")
        .AsDuplicable();
    AddInput("WeightList",
             "(vector<Tensor>), stores weight and bias data when the weight "
             "use the list format. ")
        .AsDuplicable();
    AddInput("SequenceLength",
             "(Tensor) When the input data is padding, "
             "set this parameter. This parameter represents "
             "the variable sequence lengths in a batch. "
             "The size of the vector has to equal the batch_size.")
        .AsDispensable();
    AddOutput("DropoutState",
              "Store the global drop state when training, needed by cudnn rnn.")
        .AsDispensable();
    // maybe need add intermediate outputs for cpu kernel
    AddOutput("Reserve",
              "(Tensor, a temporary output Tensor to store the reserve_data "
              "of cudnn kernel.")
        .AsIntermediate();
    AddOutput("Out",
              "(Tensor) the hidden state of LSTM operator. "
              "The shape is ( seq_len x batch_size x hidden_size) if "
              "is_bidirec is False"
              "and When is_bidirec is True, the shape will be ( seq_len x "
              "batch_size x hidden_size * 2) ");
    AddOutput("State",
              "(Tensor) the hidden state of the last step. "
              "The shape is ( num_layers x batch_size x hidden_size) if "
              "is_bidirec is False"
              "and When is_bidirec is True, the shape will be (num_layers*2 x "
              "batch_size x hidden_size)")
        .AsDuplicable();
    AddAttr<float>(
        "dropout_prob",
        "dropout prob of the dropout op"
        "the dropout ONLY work between rnn layers, not between time steps"
        "There is no dropout work on the Out tensor")
        .SetDefault(0.0);
    AddAttr<bool>("is_bidirec", "whether it is bidirectional rnn")
        .SetDefault(false);
    AddAttr<int>("input_size", "input size ot the Input Tensor").SetDefault(10);
    AddAttr<int>("hidden_size", "hidden size of rnn").SetDefault(100);
    AddAttr<int>("num_layers", "the total layer number").SetDefault(1);
    AddAttr<std::string>(
        "mode",
        "(string) rnn types, including: LSTM, GRU, RNN_RELU, RNN_TANH.");
    AddAttr<int>("seed", "seed to used if fix_seed is True").SetDefault(0);
    AddAttr<bool>("is_test", "True if in test phase.")
        .SetDefault(false)
        .AsExtra();
    AddComment(R"DOC(
)DOC");
  }
};

class RNNGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Input"), "Input", "Input", "RNN");
    OP_INOUT_CHECK(ctx->HasInputs("PreState"), "Input", "PreState", "RNN");
    OP_INOUT_CHECK(ctx->HasInput("Out"), "Input", "Out", "RNN");
    // OP_INOUT_CHECK(ctx->HasInputs("State"), "Input", "State", "RNN");

    auto SetOutGradDim = [&ctx](const std::string& name) {
      auto g_name = framework::GradVarName(name);
      if (ctx->HasOutput(g_name)) {
        ctx->SetOutputDim(g_name, ctx->GetInputDim(name));
      }
    };

    SetOutGradDim("Input");
    if (ctx->HasOutputs(framework::GradVarName("WeightList"))) {
      ctx->SetOutputsDim(framework::GradVarName("WeightList"),
                         ctx->GetInputsDim("WeightList"));
    }
    if (ctx->HasOutputs(framework::GradVarName("PreState"))) {
      ctx->SetOutputsDim(framework::GradVarName("PreState"),
                         ctx->GetInputsDim("PreState"));
    }
  }
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.device_context());
  }
};

template <typename T>
class RNNGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("rnn_grad");
    op->SetInput("Input", this->Input("Input"));
    op->SetInput("PreState", this->Input("PreState"));
    op->SetInput("WeightList", this->Input("WeightList"));
    if (this->HasInput("SequenceLength")) {
      op->SetInput("SequenceLength", this->Input("SequenceLength"));
    }
    op->SetInput("DropoutState", this->Output("DropoutState"));
    op->SetInput("Reserve", this->Output("Reserve"));
    op->SetInput("Out", this->Output("Out"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetInput(framework::GradVarName("State"), this->OutputGrad("State"));

    op->SetOutput(framework::GradVarName("WeightList"),
                  this->InputGrad("WeightList", false));

    op->SetOutput(framework::GradVarName("Input"), this->InputGrad("Input"));
    op->SetOutput(framework::GradVarName("PreState"),
                  this->InputGrad("PreState", false));
    op->SetAttrMap(this->Attrs());
  }
};

template <typename T>
class NotImpleKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_THROW(platform::errors::Unimplemented(
        "CPU is not support for this kernel now. Will be add in the future"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
DECLARE_INFER_SHAPE_FUNCTOR(rnn,
                            RnnInferShapeFunctor,
                            PD_INFER_META(phi::RnnInferMeta));

REGISTER_OPERATOR(rnn,
                  ops::RNNOp,
                  ops::RNNOpMaker,
                  ops::RNNGradOpMaker<paddle::framework::OpDesc>,
                  ops::RNNGradOpMaker<paddle::imperative::OpBase>,
                  RnnInferShapeFunctor);
REGISTER_OPERATOR(rnn_grad, ops::RNNGradOp);
