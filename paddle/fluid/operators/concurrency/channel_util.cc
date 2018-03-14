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

#include "channel_util.h"
#include "paddle/fluid/framework/var_type.h"

namespace poc = paddle::operators::concurrency;

bool poc::ChannelSend(framework::ChannelHolder *ch, framework::Variable *var) {
  auto type = framework::ToVarType(var->Type());
  if (type == framework::proto::VarType_Type_LOD_TENSOR)
    return ch->Send(var->GetMutable<framework::LoDTensor>());
  else if (type == framework::proto::VarType_Type_LOD_RANK_TABLE)
    return ch->Send(var->GetMutable<framework::LoDRankTable>());
  else if (type == framework::proto::VarType_Type_LOD_TENSOR_ARRAY)
    return ch->Send(var->GetMutable<framework::LoDTensorArray>());
  else if (type == framework::proto::VarType_Type_SELECTED_ROWS)
    return ch->Send(var->GetMutable<framework::SelectedRows>());
  else if (type == framework::proto::VarType_Type_READER)
    return ch->Send(var->GetMutable<framework::ReaderHolder>());
  else if (type == framework::proto::VarType_Type_CHANNEL)
    return ch->Send(var->GetMutable<framework::ChannelHolder>());
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
