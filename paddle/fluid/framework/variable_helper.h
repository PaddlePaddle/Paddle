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

#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/lod_tensor_array.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/framework/variable.h"

namespace paddle {
namespace framework {
void InitializeVariable(Variable* var, proto::VarType::Type var_type);

template <typename Visitor>
size_t VarSizeInBytes(Variable* var, proto::VarType::Type var_type,
                      Visitor visitor) {
  size_t ret = 0;
  switch (var_type) {
    case proto::VarType::LOD_TENSOR:
      ret = visitor.template apply<LoDTensor>(var);
      break;
    case proto::VarType::SELECTED_ROWS:
      ret = visitor.template apply<SelectedRows>(var);
      break;
    case proto::VarType::LOD_TENSOR_ARRAY:
      ret = visitor.template apply<LoDTensorArray>(var);
    default:
      break;
  }
  return ret;
}

struct VarSizeVisitor {
  template <typename T>
  size_t apply(Variable* var) {
    return 0;
  }
};

template <>
size_t VarSizeVisitor::apply<LoDTensor>(Variable* var) {
  return var->Get<LoDTensor>().memory_size();
}

template <>
size_t VarSizeVisitor::apply<SelectedRows>(Variable* var) {
  return var->Get<SelectedRows>().value().memory_size();
}

template <>
size_t VarSizeVisitor::apply<LoDTensorArray>(Variable* var) {
  auto& array = var->Get<LoDTensorArray>();
  size_t ret = 0;
  for (auto& v : array) ret += v.memory_size();
  return ret;
}
}  // namespace framework
}  // namespace paddle
