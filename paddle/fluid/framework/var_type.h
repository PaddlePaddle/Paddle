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
#include "paddle/fluid/framework/lod_rank_table.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/lod_tensor_array.h"
#include "paddle/fluid/framework/reader.h"
#include "paddle/fluid/framework/selected_rows_utils.h"
#include "paddle/fluid/framework/var_type_traits.h"
#include "paddle/fluid/framework/variable.h"

namespace paddle {
namespace framework {

template <typename T>
inline bool IsType(const std::type_index& type) {
  return type == typeid(T);
}

inline proto::VarType::Type ToVarType(int type) {
  switch (type) {
    case proto::VarType::LOD_TENSOR:
    case proto::VarType::SELECTED_ROWS:
    case proto::VarType::SPARSE_COO:
    case proto::VarType::LOD_RANK_TABLE:
    case proto::VarType::LOD_TENSOR_ARRAY:
    case proto::VarType::FETCH_LIST:
    case proto::VarType::READER:
      return static_cast<proto::VarType::Type>(type);
    default:
      PADDLE_THROW(platform::errors::Unavailable(
          "ToVarType method Unsupported type %d.", type));
  }
}

template <typename Visitor>
inline void VisitVarType(const framework::Variable& var, Visitor visitor) {
  switch (var.Type()) {
    case proto::VarType::LOD_TENSOR:
      visitor(var.Get<LoDTensor>());
      return;
    case proto::VarType::LOD_RANK_TABLE:
      visitor(var.Get<LoDRankTable>());
      return;
    case proto::VarType::LOD_TENSOR_ARRAY:
      visitor(var.Get<LoDTensorArray>());
      return;
    case proto::VarType::SELECTED_ROWS:
      visitor(var.Get<phi::SelectedRows>());
      return;
    case proto::VarType::SPARSE_COO:
      visitor(var.Get<phi::SparseCooTensor>());
      return;
    case proto::VarType::READER:
      visitor(var.Get<ReaderHolder>());
      return;
    case proto::VarType::FETCH_LIST:
      visitor(var.Get<FetchList>());
      return;
    default:
      PADDLE_THROW(platform::errors::Unavailable("Not supported visit type %s.",
                                                 ToTypeName(var.Type())));
  }
}

}  // namespace framework
}  // namespace paddle
