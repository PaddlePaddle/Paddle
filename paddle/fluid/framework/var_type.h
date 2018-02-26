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

#pragma once
#include "paddle/fluid/framework/channel.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/lod_rank_table.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/lod_tensor_array.h"
#include "paddle/fluid/framework/reader.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/framework/variable.h"

namespace paddle {
namespace framework {
inline proto::VarType::Type ToVarType(std::type_index type) {
  if (type.hash_code() == typeid(LoDTensor).hash_code()) {
    return proto::VarType_Type_LOD_TENSOR;
  } else if (type.hash_code() == typeid(LoDRankTable).hash_code()) {
    return proto::VarType_Type_LOD_RANK_TABLE;
  } else if (type.hash_code() == typeid(LoDTensorArray).hash_code()) {
    return proto::VarType_Type_LOD_TENSOR_ARRAY;
  } else if (type.hash_code() == typeid(SelectedRows).hash_code()) {
    return proto::VarType_Type_SELECTED_ROWS;
  } else if (type.hash_code() == typeid(ReaderHolder).hash_code()) {
    return proto::VarType_Type_READER;
  } else if (type.hash_code() == typeid(ChannelHolder).hash_code()) {
    return proto::VarType_Type_CHANNEL;
  } else {
    PADDLE_THROW("ToVarType:Unsupported type %s", type.name());
  }
}

template <typename Visitor>
inline void VisitVarType(const framework::Variable& var, Visitor visitor) {
  switch (ToVarType(var.Type())) {
    case proto::VarType_Type_LOD_TENSOR:
      visitor(var.Get<LoDTensor>());
      return;
    case proto::VarType_Type_LOD_RANK_TABLE:
      visitor(var.Get<LoDRankTable>());
      return;
    case proto::VarType_Type_LOD_TENSOR_ARRAY:
      visitor(var.Get<LoDTensorArray>());
      return;
    case proto::VarType_Type_SELECTED_ROWS:
      visitor(var.Get<SelectedRows>());
      return;
    case proto::VarType_Type_READER:
      visitor(var.Get<ReaderHolder>());
      return;
    case proto::VarType_Type_CHANNEL:
      visitor(var.Get<ChannelHolder>());
      return;
    default:
      PADDLE_THROW("Not supported visit type, %d", ToVarType(var.Type()));
  }
}

}  // namespace framework
}  // namespace paddle
