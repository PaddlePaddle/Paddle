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
static constexpr char Status[] = "Status";
static constexpr char Out[] = "Out";

namespace paddle {
namespace operators {

void SetReceiveStatus(const platform::Place &dev_place,
                      framework::Variable *status_var, bool status) {
  auto cpu = platform::CPUPlace();
  auto status_tensor =
      status_var->GetMutable<framework::LoDTensor>()->mutable_data<bool>({1},
                                                                         cpu);
  status_tensor[0] = status;
}

class ChannelRecvOp : public framework::OperatorBase {
 public:
  ChannelRecvOp(const std::string &type,
                const framework::VariableNameMap &inputs,
                const framework::VariableNameMap &outputs,
                const framework::AttributeMap &attrs)
      : framework::OperatorBase(type, inputs, outputs, attrs) {}

  void InferShape(framework::InferShapeContext *ctx) const {
    PADDLE_ENFORCE(ctx->HasInput(Channel),
                   "Input(Channel) of ChannelRecvOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput(Out),
                   "Input(Channel) of ChannelRecvOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput(Status),
                   "Output(Status) of ChannelRecvOp should not be null.");
    ctx->SetOutputDim("Status", {1});
  }

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &dev_place) const override {
    // Get the channel holder created by channel_create op, passed as input.
    framework::ChannelHolder *ch =
        scope.FindVar(Input(Channel))->GetMutable<framework::ChannelHolder>();
    auto output_var = scope.FindVar(Output(Out));
    // Receive the data from the channel.
    bool ok = concurrency::ChannelReceive(ch, output_var);

    // Set the status output of the `ChannelReceive` call.
    SetReceiveStatus(dev_place, scope.FindVar(Output(Status)), ok);
  }
};

class ChannelRecvOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  ChannelRecvOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput(Channel,
             "(Channel) A variable which \"receives\" the a value sent"
             "to it by a channel_send op.")
        .AsDuplicable();
    AddOutput(Out,
              "(Variable) Output Variable that will hold the data received"
              " from the Channel")
        .AsDuplicable();
    AddOutput(Status,
              "(Tensor) An LoD Tensor that returns a boolean status of the"
              "result of the receive operation.")
        .AsDuplicable();
    AddComment(R"DOC(
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OPERATOR(channel_recv, paddle::operators::ChannelRecvOp,
                  paddle::framework::EmptyGradOpMaker,
                  paddle::operators::ChannelRecvOpMaker);
