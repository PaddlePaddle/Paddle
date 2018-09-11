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

#include "paddle/fluid/operators/concurrency/channel_util.h"
#include "paddle/fluid/framework/var_type.h"

namespace poc = paddle::operators::concurrency;

void poc::ChannelSend(framework::ChannelHolder *ch, framework::Variable *var) {
  auto type = framework::ToVarType(var->Type());
  if (type == framework::proto::VarType_Type_LOD_TENSOR)
    ch->Send(var->GetMutable<framework::LoDTensor>());
  else if (type == framework::proto::VarType_Type_LOD_RANK_TABLE)
    ch->Send(var->GetMutable<framework::LoDRankTable>());
  else if (type == framework::proto::VarType_Type_LOD_TENSOR_ARRAY)
    ch->Send(var->GetMutable<framework::LoDTensorArray>());
  else if (type == framework::proto::VarType_Type_SELECTED_ROWS)
    ch->Send(var->GetMutable<framework::SelectedRows>());
  else if (type == framework::proto::VarType_Type_READER)
    ch->Send(var->GetMutable<framework::ReaderHolder>());
  else if (type == framework::proto::VarType_Type_CHANNEL)
    ch->Send(var->GetMutable<framework::ChannelHolder>());
  else
    PADDLE_THROW("ChannelSend:Unsupported type");
}

bool poc::ChannelReceive(framework::ChannelHolder *ch,
                         framework::Variable *var) {
  // Get type of channel and use that to call mutable data for Variable
  auto type = framework::ToVarType(ch->Type());
  if (type == framework::proto::VarType_Type_LOD_TENSOR)
    return ch->Receive(var->GetMutable<framework::LoDTensor>());
  else if (type == framework::proto::VarType_Type_LOD_RANK_TABLE)
    return ch->Receive(var->GetMutable<framework::LoDRankTable>());
  else if (type == framework::proto::VarType_Type_LOD_TENSOR_ARRAY)
    return ch->Receive(var->GetMutable<framework::LoDTensorArray>());
  else if (type == framework::proto::VarType_Type_SELECTED_ROWS)
    return ch->Receive(var->GetMutable<framework::SelectedRows>());
  else if (type == framework::proto::VarType_Type_READER)
    return ch->Receive(var->GetMutable<framework::ReaderHolder>());
  else if (type == framework::proto::VarType_Type_CHANNEL)
    return ch->Receive(var->GetMutable<framework::ChannelHolder>());
  else
    PADDLE_THROW("ChannelReceive:Unsupported type");
}

void poc::ChannelAddToSendQ(framework::ChannelHolder *ch, const void *referrer,
                            framework::Variable *var,
                            std::shared_ptr<std::condition_variable_any> cond,
                            std::function<bool(framework::ChannelAction)> cb) {
  auto type = framework::ToVarType(var->Type());
  if (type == framework::proto::VarType_Type_LOD_TENSOR) {
    ch->AddToSendQ(referrer, var->GetMutable<framework::LoDTensor>(), cond, cb);
  } else if (type == framework::proto::VarType_Type_LOD_RANK_TABLE) {
    ch->AddToSendQ(referrer, var->GetMutable<framework::LoDRankTable>(), cond,
                   cb);
  } else if (type == framework::proto::VarType_Type_LOD_TENSOR_ARRAY) {
    ch->AddToSendQ(referrer, var->GetMutable<framework::LoDTensorArray>(), cond,
                   cb);
  } else if (type == framework::proto::VarType_Type_SELECTED_ROWS) {
    ch->AddToSendQ(referrer, var->GetMutable<framework::SelectedRows>(), cond,
                   cb);
  } else if (type == framework::proto::VarType_Type_READER) {
    ch->AddToSendQ(referrer, var->GetMutable<framework::ReaderHolder>(), cond,
                   cb);
  } else if (type == framework::proto::VarType_Type_CHANNEL) {
    ch->AddToSendQ(referrer, var->GetMutable<framework::ChannelHolder>(), cond,
                   cb);
  } else {
    PADDLE_THROW("ChannelAddToSendQ:Unsupported type");
  }
}

void poc::ChannelAddToReceiveQ(
    framework::ChannelHolder *ch, const void *referrer,
    framework::Variable *var, std::shared_ptr<std::condition_variable_any> cond,
    std::function<bool(framework::ChannelAction)> cb) {
  auto type = framework::ToVarType(var->Type());
  if (type == framework::proto::VarType_Type_LOD_TENSOR) {
    ch->AddToReceiveQ(referrer, var->GetMutable<framework::LoDTensor>(), cond,
                      cb);
  } else if (type == framework::proto::VarType_Type_LOD_RANK_TABLE) {
    ch->AddToReceiveQ(referrer, var->GetMutable<framework::LoDRankTable>(),
                      cond, cb);
  } else if (type == framework::proto::VarType_Type_LOD_TENSOR_ARRAY) {
    ch->AddToReceiveQ(referrer, var->GetMutable<framework::LoDTensorArray>(),
                      cond, cb);
  } else if (type == framework::proto::VarType_Type_SELECTED_ROWS) {
    ch->AddToReceiveQ(referrer, var->GetMutable<framework::SelectedRows>(),
                      cond, cb);
  } else if (type == framework::proto::VarType_Type_READER) {
    ch->AddToReceiveQ(referrer, var->GetMutable<framework::ReaderHolder>(),
                      cond, cb);
  } else if (type == framework::proto::VarType_Type_CHANNEL) {
    ch->AddToReceiveQ(referrer, var->GetMutable<framework::ChannelHolder>(),
                      cond, cb);
  } else {
    PADDLE_THROW("ChannelAddToReceiveQ:Unsupported type");
  }
}
