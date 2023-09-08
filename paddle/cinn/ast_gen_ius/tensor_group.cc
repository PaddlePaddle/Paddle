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
    std::set<ir::Expr> used_tensors = ir::CollectIRNodesWithoutTensor(
        tensor->body(), [](const Expr* x) { return x->as_tensor(); });
    for (const Expr& x : used_tensors) {
      const ir::Tensor to_dep = x.as_tensor_ref();
      all_tensors.insert(to_dep);
      this->CtrlDepend(tensor, to_dep);
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

std::vector<ir::Tensor> TensorGroup::GetGenFuncTopoOrder(
    const std::vector<ir::Tensor>& func_args) {
  std::unordered_map<std::string, int> in_degree;
  for (const auto& dep_pair : ctrl_dep_) {
    const std::unordered_set<std::string>& dep_tensor_names = dep_pair.second;
    for (const std::string& name : dep_tensor_names) {
      if (in_degree.count(name)) {
        ++in_degree[name];
      } else {
        in_degree[name] = 1;
      }
    }
  }

  std::vector<ir::Tensor> ret;
  std::vector<std::string> stack;
  for (const auto& name_tensor : name_to_tensor_) {
    if (!in_degree.count(name_tensor.first)) {
      stack.emplace_back(name_tensor.first);
    }
  }

  while (!stack.empty()) {
    const std::string& cur = stack.back();

    bool name_in_args = false;
    for (const ir::Tensor& arg : func_args) {
      if (cur == arg->name) {
        name_in_args = true;
      }
    }
    if (!name_in_args) {
      ret.push_back(name_to_tensor_[cur]);
    }

    if (ctrl_dep_.count(cur)) {
      for (const std::string& name : ctrl_dep_[cur]) {
        --in_degree[name];
        if (in_degree[name] == 0) {
          stack.emplace_back(name);
        }
      }
    }
    stack.pop_back();
  }
  return ret;
}

bool TensorGroup::HasMarkedReduceInit(const std::string& tensor_name) const {
  return tensor_name_needs_reduce_init_.count(tensor_name);
}

ir::Tensor TensorGroup::MarkReduceInit(const std::string& tensor_name) {
  // TODO(zhhsplendid): add check
  tensor_name_needs_reduce_init_.insert(tensor_name);
}

void TensorGroup::CtrlDepend(const ir::Tensor& tensor,
                             const ir::Tensor& to_dep) {
  ctrl_dep_[tensor->name].insert(to_dep->name);
  if (!name_to_tensor_.count(to_dep->name)) {
    name_to_tensor_[to_dep->name] = to_dep;
  }
}

std::set<ir::Tensor> TensorGroup::GetCrtlDepTensors(
    const std::string& tensor_name) {
  if (!ctrl_dep_.count(tensor_name)) {
    return {};
  }
  std::set<ir::Tensor> ret;
  for (const std::string& dep_name : ctrl_dep_[tensor_name]) {
    ret.insert(name_to_tensor_[dep_name]);
  }
  return ret;
}

std::string TensorGroup::GetShareMemRootName(const std::string& tensor_name) {
  if (!share_memory_tensor_.count(tensor_name)) {
    share_memory_tensor_[tensor_name] = tensor_name;
    return tensor_name;
  }
  if (share_memory_tensor_[tensor_name] == tensor_name) {
    return tensor_name;
  }
  share_memory_tensor_[tensor_name] =
      GetShareMemRootName(share_memory_tensor_[tensor_name]);
  return share_memory_tensor_[tensor_name];
}

void TensorGroup::ShareMemoryBuffer(const ir::Tensor& tensor,
                                    const ir::Tensor& to_share) {
  share_memory_tensor_[GetShareMemRootName(to_share->name)] =
      GetShareMemRootName(tensor->name);
}

absl::flat_hash_map<std::string, ir::Tensor> TensorGroup::AllocateBuffers() {
  std::unordered_set<std::string> allocated_roots;
  for (auto& name_tensor : name_to_tensor_) {
    std::string root_name = GetShareMemRootName(name_tensor.first);

    // Allocate root buffer
    if (!allocated_roots.count(root_name)) {
      ir::Tensor root_tensor = name_to_tensor_[root_name];
      if (!root_tensor->buffer.defined() && !root_tensor->type().is_void()) {
        root_tensor->WithBuffer();
      }
      allocated_roots.insert(root_name);
    }

    // Share buffer
    if (root_name != name_tensor.first) {
      ir::Tensor& root_tensor = name_to_tensor_[root_name];
      ir::Tensor& tensor = name_tensor.second;

      auto keep_shape = root_tensor->buffer->shape;
      tensor->Bind(root_tensor->buffer);
      root_tensor->buffer->shape = keep_shape;
      tensor->buffer->shape = keep_shape;
    }
  }

  return name_to_tensor_;
}

}  // namespace ast_gen_ius
}  // namespace cinn
