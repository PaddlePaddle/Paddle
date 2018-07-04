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
#include <paddle/fluid/framework/lod_rank_table.h>
#include <paddle/fluid/framework/lod_tensor_array.h>
#include <paddle/fluid/framework/reader.h>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/var_type.h"
#include "paddle/fluid/operators/concurrency/channel_util.h"
#include "paddle/fluid/operators/math/math_function.h"

static constexpr char Channel[] = "Channel";
static constexpr char X[] = "X";

namespace paddle {
namespace operators {

class ChannelSendOp : public framework::OperatorBase {
 public:
  ChannelSendOp(const std::string &type,
                const framework::VariableNameMap &inputs,
                const framework::VariableNameMap &outputs,
                const framework::AttributeMap &attrs)
      : framework::OperatorBase(type, inputs, outputs, attrs) {}

  void InferShape(framework::InferShapeContext *ctx) const {
    PADDLE_ENFORCE(ctx->HasInput(Channel),
                   "Input(Channel) of ChannelSendOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput(X),
                   "Input(X) of ChannelSendOp should not be null.");
  }

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &dev_place) const override {
    // Get the channel holder created by channel_create op, passed as input.
    framework::ChannelHolder *ch =
        scope.FindVar(Input(Channel))->GetMutable<framework::ChannelHolder>();
    auto input_var = scope.FindVar(Input(X));

    // Send the input data through the channel.
    concurrency::ChannelSend(ch, input_var);
  }
};

class ChannelSendOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput(Channel,
             "(Channel) A variable which \"sends\" the passed in value to "
             "a listening receiver.")
        .AsDuplicable();
    AddInput(X, "(Variable) The value which gets sent by the channel.")
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
