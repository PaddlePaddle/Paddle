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
#include "paddle/cinn/ir/utils/ir_nodes_collector.h"

namespace cinn {
namespace ast_gen_ius {

TensorGroup::TensorGroup(const std::vector<ir::Tensor>& tensors) {
  std::set<ir::Tensor> all_tensors(tensors.begin(), tensors.end());

  for (auto& tensor : tensors) {
    auto used_tensors = ir::CollectIRNodes(
        tensor->body(), [](const Expr* x) { return x->as_tensor(); });
    for (const Expr& x : used_tensors) {
      all_tensors.insert(x.as_tensor_ref());
    }
  }

  for (const ir::Tensor& t : all_tensors) {
    name_to_tensor_.insert({t->name, t});
  }
}

TensorGroup::~TensorGroup() {}

bool TensorGroup::Contain(const std::string& name) const {
  return name_to_tensor_.find(name) != name_to_tensor_.end();
}

void TensorGroup::Insert(const ir::Tensor& tensor) {
  name_to_tensor_.insert({tensor->name, tensor});
}

ir::Tensor TensorGroup::Get(const std::string& name) {
  return name_to_tensor_[name];
}

std::set<ir::Tensor> TensorGroup::GetAllTensors() {
  std::set<ir::Tensor> all_tensors;
  for (const std::pair<std::string, ir::Tensor>& p : name_to_tensor_) {
    all_tensors.insert(p.second);
  }
  return all_tensors;
}

bool HasMarkedReduceInit(const ir::_Tensor_& tensor) const {
  return tensor_name_needs_reduce_init_.find(tensor.name) !=
         tensor_name_needs_reduce_init_.end();
}

ir::Tensor TensorGroup::MarkReduceInit(const ir::_Tensor_& tensor) {
  // TODO(zhhsplendid): add check
  tensor_name_needs_reduce_init_.insert(tensor.name);
}

void TensorGroup::CtrlDepend(const ir::Tensor& tensor,
                             const ir::Tensor& to_dep) {
  ctrl_dep_[tensor->name].insert(to_dep->name);
  if (!name_to_tensor_.count(to_dep->name)) {
    name_to_tensor_[to_dep->name] = to_dep;
  }
}

std::set<ir::Tensor> GetCrtlDepTensors(const std::string& tensor_name) {
  std::set<ir::Tensor> ret;
  for (const std::string& dep_name : ctrl_dep_[tensor]) {
    ret.insert(name_to_tensor_[dep_name]);
  }
  return ret;
}

}  // namespace ast_gen_ius
}  // namespace cinn
