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
#include "paddle/cinn/poly/stage.h"

namespace cinn {
namespace ast_gen_ius {

TensorGroup::TensorGroup(const std::vector<ir::Tensor>& tensors) {
  for (const ir::Tensor& tensor : tensors) {
    output_tensor_names_.insert(tensor->name);
    this->Insert(tensor);
  }
}

void TensorGroup::ShowLog() const {
  VLOG(6) << "Showing log for TensorGroup";
  for (auto& p : name_to_tensor_) {
    VLOG(6) << "Tensor name = " << p.first << " depends on {";
    if (ctrl_dep_.count(p.first)) {
      for (auto& dep_name : ctrl_dep_.at(p.first)) {
        VLOG(6) << dep_name;
      }
    }
    VLOG(6) << "}";
  }
}

TensorGroup::TensorGroup(
    const std::unordered_map<std::string, ir::Tensor>& tensor_map) {
  for (const auto& map_pair : tensor_map) {
    const ir::Tensor& tensor = map_pair.second;
    output_tensor_names_.insert(tensor->name);
    this->Insert(tensor);
  }
}

TensorGroup::~TensorGroup() {}

bool TensorGroup::Contain(const std::string& name) const {
  return name_to_tensor_.find(name) != name_to_tensor_.end();
}

void TensorGroup::Insert(const ir::Tensor& tensor) {
  if (!name_to_tensor_.count(tensor->name)) {
    name_to_tensor_.insert({tensor->name, tensor});
  }

  // Using set to de-duplicate
  std::set<ir::Tensor> dep_tensors;
  std::set<ir::Expr> used_tensors = ir::ir_utils::CollectIRNodes(
      tensor->body(), [](const Expr* x) { return x->as_tensor(); });
  for (const Expr& x : used_tensors) {
    const ir::Tensor to_dep = x.as_tensor_ref();
    dep_tensors.insert(to_dep);
    this->CtrlDepend(tensor, to_dep);
  }

  for (const ir::Tensor& t : dep_tensors) {
    this->Insert(t);
  }
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
    in_degree[dep_pair.first] = dep_tensor_names.size();
    VLOG(6) << "indegree[" << dep_pair.first
            << "] = " << dep_tensor_names.size();
  }

  std::vector<ir::Tensor> ret;

  // Using set instead of vector/stack in order to get fix alpha-beta order topo
  std::set<std::string> node_set;
  for (const auto& name_tensor : name_to_tensor_) {
    if (!in_degree.count(name_tensor.first)) {
      node_set.insert(name_tensor.first);
    }
  }

  std::set<std::string> input_arg_names;
  for (const ir::Tensor& arg : func_args) {
    input_arg_names.insert(arg->name);
  }
  for (const std::string& name : output_tensor_names_) {
    input_arg_names.erase(name);
  }

  while (!node_set.empty()) {
    const std::string cur = *(node_set.begin());
    node_set.erase(node_set.begin());
    if (!input_arg_names.count(cur)) {
      ret.push_back(name_to_tensor_[cur]);
    }

    for (const auto& dep_pair : ctrl_dep_) {
      const std::unordered_set<std::string>& dep_tensor_names = dep_pair.second;
      if (dep_tensor_names.count(cur)) {
        --in_degree[dep_pair.first];
        if (in_degree[dep_pair.first] == 0) {
          node_set.insert(dep_pair.first);
        }
      }
    }
  }
  return ret;
}

void TensorGroup::CtrlDepend(const ir::Tensor& tensor,
                             const ir::Tensor& to_dep) {
  ctrl_dep_[tensor->name].insert(to_dep->name);
  if (!name_to_tensor_.count(tensor->name)) {
    name_to_tensor_[tensor->name] = tensor;
  }
  if (!name_to_tensor_.count(to_dep->name)) {
    name_to_tensor_[to_dep->name] = to_dep;
  }
}

std::set<ir::Tensor> TensorGroup::GetCtrlDepTensors(
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

void TensorGroup::MarkShareMemBuffer(const ir::Tensor& tensor,
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
        VLOG(6) << "Bind root_tensor " << root_name << " with buffer "
                << root_tensor->buffer->name;
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
      VLOG(6) << "Share buffer " << root_name << " with " << name_tensor.first;
    }
  }

  return name_to_tensor_;
}

}  // namespace ast_gen_ius
}  // namespace cinn
