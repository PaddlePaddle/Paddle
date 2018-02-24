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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/channel.h"

namespace paddle {
namespace operators {

static constexpr char Channel[] = "Channel";
static constexpr char Val[] = "Val";
static constexpr char Status[] = "Status";

class ChannelSendOp : public framework::OperatorBase {
public:
    ChannelSendOp(const std::string &type,
                const framework::VariableNameMap &inputs,
                const framework::VariableNameMap &outputs,
                const framework::AttributeMap &attrs)
            : framework::OperatorBase(type, inputs, outputs, attrs) {}

    void InferShape(framework::InferShapeContext* ctx) const {
        PADDLE_ENFORCE(ctx->HasInput("Channel"),
                       "Input(Channel) of ChannelSendOp should not be null.");
        PADDLE_ENFORCE(ctx->HasInput("Val"),
                       "Input(Val) of ChannelSendOp should not be null.");
        PADDLE_ENFORCE(ctx->HasOutput("Status"),
                       "Output(Status) of ChannelSendOp should not be null.");
        ctx->SetOutputDim("Status", {1});
    }

private:
    void RunImpl(const framework::Scope &scope,
                 const platform::Place &dev_place) const override {
    }
};

class ChannelSendOpMaker : public framework::OpProtoAndCheckerMaker {
public:
    ChannelSendOpMaker(OpProto *proto, OpAttrChecker *op_checker)
            : OpProtoAndCheckerMaker(proto, op_checker) {
      AddInput(Channel,
               "(Channel) A variable which \"sends\" the passed in value to "
                       "a listening receiver.")
              .AsDuplicable();
      AddInput(Val,
                 "(Variable) The value which gets sent by the channel.")
                .AsDuplicable();
      AddOutput(Status,
                "(Tensor) An LoD Tensor that returns a boolean status of the"
                        "result of the send operation.")
              .AsDuplicable();
      AddComment(R"DOC(
)DOC");
    }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OPERATOR(channel_send, paddle::operators::ChannelSendOp,
        paddle::framework::EmptyGradOpMaker,
        paddle::operators::ChannelSendOpMaker);
