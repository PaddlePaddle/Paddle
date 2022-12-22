// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/eager/api/prims/composite_grad_desc_maker.h"
#include "paddle/fluid/eager/api/prims/utils.h"
#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/phi/core/macros.h"
namespace paddle {
namespace prims {

template <>
framework::VarDesc* CreateVar(const std::string& name,
                              std::vector<int64_t> shape,
                              bool is_persistable,
                              framework::proto::VarType::Type data_type) {
  auto* var = StaticCompositeContext::Instance().GetBlock()->Var(name);
  var->SetType(framework::proto::VarType::LOD_TENSOR);
  var->SetDataType(data_type);
  var->SetShape(shape);
  var->SetPersistable(is_persistable);
  return var;
}

template <>
framework::VarDesc* CreateVarLike(const framework::VarDesc& var) {
  return CreateVar<framework::VarDesc>(
      var.Name(), var.GetShape(), var.IsParameter(), var.GetDataType());
}

}  // namespace prims
}  // namespace paddle
