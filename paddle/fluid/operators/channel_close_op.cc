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

#include "paddle/fluid/framework/channel.h"
#include "paddle/fluid/framework/op_registry.h"

namespace pf = paddle::framework;
static constexpr char kChannel[] = "Channel";

namespace paddle {
namespace operators {

class ChannelCloseOp : public framework::OperatorBase {
 public:
  ChannelCloseOp(const std::string &type,
                 const framework::VariableNameMap &inputs,
                 const framework::VariableNameMap &outputs,
                 const framework::AttributeMap &attrs)
      : framework::OperatorBase(type, inputs, outputs, attrs) {}

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &dev_place) const override {
    auto &inp = *scope.FindVar(Input(kChannel));

    // Get the mutable version of the channel variable and closes it.
    pf::ChannelHolder *ch = inp.GetMutable<framework::ChannelHolder>();
    ch->close();
  }
};

class ChannelCloseOpOpInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *context) const override {
    PADDLE_ENFORCE(context->HasInput("Channel"),
                   "The input of ChannelClose op must be set");
  }
};

class ChannelCloseOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput(kChannel,
             "The Channel Variable that should be closed by"
             " the ChannelClose Op.");
    AddComment(R"DOC(
Channel Close Operator.

This operator closes an open channel.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OPERATOR(channel_close, paddle::operators::ChannelCloseOp,
                  paddle::framework::EmptyGradOpMaker,
                  paddle::operators::ChannelCloseOpMaker);
