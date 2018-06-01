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
#include "paddle/fluid/framework/lod_rank_table.h"
#include "paddle/fluid/framework/lod_tensor_array.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/reader.h"

namespace pf = paddle::framework;

static constexpr char kOutput[] = "Out";

namespace paddle {
namespace operators {

class ChannelCreateOp : public framework::OperatorBase {
 public:
  ChannelCreateOp(const std::string &type,
                  const framework::VariableNameMap &inputs,
                  const framework::VariableNameMap &outputs,
                  const framework::AttributeMap &attrs)
      : framework::OperatorBase(type, inputs, outputs, attrs) {}

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &dev_place) const override {
    auto &out = *scope.FindVar(Output(kOutput));

    // Determine the datatype and capacity of the channel to be created
    // from the attributes provided.
    auto dtype =
        static_cast<framework::proto::VarType::Type>(Attr<int>("data_type"));
    auto capacity = Attr<int>("capacity");

    // Based on the datatype, create a new channel holder initialized with
    // the given capacity. When capacity is 0, an unbuffered channel is
    // created.
    pf::ChannelHolder *ch = out.GetMutable<framework::ChannelHolder>();
    if (dtype == framework::proto::VarType::LOD_TENSOR) {
      ch->Reset<pf::LoDTensor>(capacity);
    } else if (dtype == framework::proto::VarType::SELECTED_ROWS) {
      ch->Reset<pf::SelectedRows>(capacity);
    } else if (dtype == framework::proto::VarType::LOD_RANK_TABLE) {
      ch->Reset<pf::LoDRankTable>(capacity);
    } else if (dtype == framework::proto::VarType::LOD_TENSOR_ARRAY) {
      ch->Reset<pf::LoDTensorArray>(capacity);
    } else if (dtype == framework::proto::VarType::READER) {
      ch->Reset<pf::ReaderHolder>(capacity);
    } else if (dtype == framework::proto::VarType::CHANNEL) {
      ch->Reset<pf::ChannelHolder>(capacity);
    } else if (dtype == framework::proto::VarType::BOOL) {
      ch->Reset<bool>(capacity);
    } else if (dtype == framework::proto::VarType::INT32) {
      ch->Reset<int>(capacity);
    } else if (dtype == framework::proto::VarType::INT64) {
      ch->Reset<int64_t>(capacity);
    } else if (dtype == framework::proto::VarType::FP32) {
      ch->Reset<float>(capacity);
    } else if (dtype == framework::proto::VarType::FP64) {
      ch->Reset<double>(capacity);
    } else {
      PADDLE_THROW(
          "Data type %d is not in "
          "[LOD_TENSOR, SELECTED_ROWS, LOD_RANK_TABLE, LOD_TENSOR_ARRAY, "
          "READER, CHANNEL, BOOL, INT32, INT64, FP32, FP64]",
          dtype);
    }
  }
};

class ChannelCreateOpOpInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *context) const override {
    PADDLE_ENFORCE(context->HasOutput(kOutput),
                   "The output of ChannelCreate op must be set");
    context->SetOutputDim(kOutput, {1});
  }
};

class ChannelCreateOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddOutput(kOutput,
              "The object of a Channel type created by ChannelCreate Op.");
    AddAttr<int>("capacity", "The size of the buffer of Channel.")
        .SetDefault(0);
    AddAttr<int>("data_type", "The data type of elements inside the Channel.");
    AddComment(R"DOC(
Channel Create Operator.

This operator creates an object of the VarType Channel and returns it.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OPERATOR(channel_create, paddle::operators::ChannelCreateOp,
                  paddle::framework::EmptyGradOpMaker,
                  paddle::operators::ChannelCreateOpMaker);
