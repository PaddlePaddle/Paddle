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
#include "paddle/framework/framework.pb.h"
#include "paddle/framework/lod_rank_table.h"
#include "paddle/framework/lod_tensor.h"
#include "paddle/framework/lod_tensor_array.h"

namespace paddle {
namespace framework {
inline VarDesc::VarType ToVarType(std::type_index type) {
  if (type.hash_code() == typeid(LoDTensor).hash_code()) {
    return VarDesc_VarType_LOD_TENSOR;
  } else if (type.hash_code() == typeid(LoDRankTable).hash_code()) {
    return VarDesc_VarType_LOD_RANK_TABLE;
  } else if (type.hash_code() == typeid(LoDTensorArray).hash_code()) {
    return VarDesc_VarType_LOD_TENSOR_ARRAY;
  } else if (type.hash_code() == typeid(SelectedRows).hash_code()) {
    return VarDesc_VarType_SELECTED_ROWS;
  } else {
    PADDLE_THROW("ToVarType:Unsupported type %s", type.name());
  }
}

template <typename Visitor>
inline void VisitVarType(const Variable& var, Visitor visitor) {
  switch (ToVarType(var.Type())) {
    case VarDesc_VarType_LOD_TENSOR:
      visitor(var.Get<framework::LoDTensor>());
      return;
    case VarDesc_VarType_LOD_RANK_TABLE:
      visitor(var.Get<LoDRankTable>());
      return;
    case VarDesc_VarType_LOD_TENSOR_ARRAY:
      visitor(var.Get<LoDTensorArray>());
      return;
    case VarDesc_VarType_SELECTED_ROWS:
      visitor(var.Get<SelectedRows>());
      return;
    default:
      PADDLE_THROW("Not supported visit type, %d", ToVarType(var.Type()));
  }
}

}  // namespace framework
}  // namespace paddle
