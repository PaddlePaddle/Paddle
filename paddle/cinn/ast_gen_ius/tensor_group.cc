// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/ast_gen_ius/tensor_group.h"

#include <unordered_map>
#include <vector>

#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/tensor.h"

namespace cinn {
namespace ast_gen_ius {

TensorGroup::TensorGroup(const std::vector<ir::Tensor>& tensors) {
  for (const ir::Tensor& t : tensors) {
    name_to_tensor_.insert({t->name, t});
  }
}

TensorGroup::~TensorGroup() {}

void TensorGroup::Insert(const ir::Tensor& tensor) {
  name_to_tensor_.insert({tensor->name, tensor});
}

ir::Tensor TensorGroup::MarkReduceInit(const ir::_Tensor_& tensor) {
  // TODO(zhhsplendid): add check
  tensor_name_needs_reduce_init_.insert(tensor.name);
}

ir::Tensor TensorGroup::Get(const std::string& name) {
  return name_to_tensor_[name];
}

void TensorGroup::CtrlDepend(const ir::Tensor& tensor,
                             const ir::Tensor& to_dep) {
  ctrl_dep_[tensor->name].insert(to_dep->name);
}

}  // namespace ast_gen_ius
}  // namespace cinn
