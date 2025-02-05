// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/pass_test_util.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <functional>
#include <iterator>
#include <list>
#include <map>

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/ir/graph_traits.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/ir/pass_tester_helper.h"
#include "paddle/fluid/framework/op_proto_maker.h"

namespace paddle::framework::ir::test {

OpDesc* CreateOp(ProgramDesc* prog,
                 const std::string& op_type_name,
                 const std::vector<InOutVarNamePair>& inputs,
                 const std::vector<InOutVarNamePair>& outputs,
                 bool use_mkldnn) {
  auto* op = prog->MutableBlock(0)->AppendOp();
  op->SetType(op_type_name);
  op->SetAttr("use_mkldnn", use_mkldnn);

  for (const auto& input : inputs) {
    op->SetInput(input.first, {input.second});
  }
  for (const auto& output : outputs) {
    op->SetOutput(output.first, {output.second});
  }

  op->SetAttr(OpProtoAndCheckerMaker::OpRoleAttrName(),
              static_cast<int>(OpRole::kForward));
  return op;
}

bool TestIsReachable(const Graph& graph, std::string from, std::string to) {
  auto hash = [](const Node* node) -> std::string {
    return node->Name() + std::to_string(node->id());
  };

  auto find_node = [&](const Graph& graph, const std::string& name) -> Node* {
    for (auto& node : GraphTraits::DFS(graph)) {
      if (name == hash(&node)) {
        return &node;
      }
    }

    return nullptr;
  };

  if (from == to) return true;

  std::map<std::string, bool> visited;
  // update the from and to strings to hashed equivs in loop from graph traits
  for (auto& node : GraphTraits::DFS(graph)) {
    auto hashed = hash(&node);
    if (node.Name() == from) {
      from = hashed;
    }
    if (node.Name() == to) {
      to = hashed;
    }
    visited[hashed] = false;
  }

  visited[from] = true;

  std::list<std::string> queue;
  queue.push_back(from);

  while (!queue.empty()) {
    auto cur = find_node(graph, queue.front());
    queue.pop_front();
    if (cur == nullptr) {
      return false;
    }

    for (auto n : cur->outputs) {
      auto hashed_name = hash(n);
      if (hashed_name == to) {
        return true;
      }

      if (!visited[hashed_name]) {
        visited[hashed_name] = true;
        queue.push_back(hashed_name);
      }
    }
  }
  return false;
}

bool AssertOpsCount(const Graph& graph,
                    std::vector<OpTypeCountPair> op_type_count) {
  for (auto* node : graph.Nodes()) {
    if (!node->IsOp()) {
      continue;
    }

    const std::string op_type_name = node->Op()->Type();
    auto op_it = std::find_if(std::begin(op_type_count),
                              std::end(op_type_count),
                              [op_type_name](const OpTypeCountPair& p) {
                                return op_type_name == p.first;
                              });
    if (op_it != std::end(op_type_count)) {
      op_it->second--;
    }
  }

  bool result{true};

  for (const OpTypeCountPair& p : op_type_count) {
    result = result && (p.second == 0);
  }
  return result;
}

ProgramDesc BuildProgramDesc(const std::vector<std::string>& transient_vars,
                             const std::vector<std::string>& persistent_vars) {
  ProgramDesc prog;

  auto add_var_to_prog = [&prog](const std::string& var_name) -> VarDesc* {
    auto var = prog.MutableBlock(0)->Var(var_name);
    var->SetType(proto::VarType::DENSE_TENSOR);
    return var;
  };

  for (const auto& v : transient_vars) {
    add_var_to_prog(v);
  }

  for (const auto& v : persistent_vars) {
    auto* var = add_var_to_prog(v);
    var->SetPersistable(true);
  }

  return prog;
}

bool RunPassAndAssert(Graph* graph,
                      const std::string& pass_name,
                      const std::string& from,
                      const std::string& to,
                      int removed_nodes_count,
                      int added_nodes_count) {
  if (!TestIsReachable(*graph, from, to)) return false;

  int original_nodes_num = static_cast<int>(graph->Nodes().size());
  auto pass = PassRegistry::Instance().Get(pass_name);
  pass->Apply(graph);
  int current_nodes_num = static_cast<int>(graph->Nodes().size());

  if (!TestIsReachable(*graph, from, to)) return false;

  int expected_nodes_num =
      original_nodes_num - removed_nodes_count + added_nodes_count;
  return expected_nodes_num == current_nodes_num;
}

template <typename T>
void InitDenseTensorHolder(const Scope& scope,
                           const phi::Place& place,
                           const std::string& var_name,
                           const std::vector<int64_t>& dims,
                           const T* data) {
  auto var = scope.FindLocalVar(var_name);
  auto tensor = var->GetMutable<phi::DenseTensor>();
  auto* tensor_mem_ptr =
      tensor->mutable_data<T>(common::make_ddim(dims), place);
  if (data != nullptr) {
    std::memcpy(tensor_mem_ptr, data, tensor->memory_size());
  } else {
    std::memset(tensor_mem_ptr, 0, tensor->memory_size());
  }
}

// Instantiate for below data types.
template void InitDenseTensorHolder<float>(const Scope&,
                                           const phi::Place&,
                                           const std::string&,
                                           const std::vector<int64_t>&,
                                           const float*);
template void InitDenseTensorHolder<int>(const Scope&,
                                         const phi::Place&,
                                         const std::string&,
                                         const std::vector<int64_t>&,
                                         const int*);
template void InitDenseTensorHolder<double>(const Scope&,
                                            const phi::Place&,
                                            const std::string&,
                                            const std::vector<int64_t>&,
                                            const double*);

OpDesc* GetOp(const ProgramDesc& prog,
              const std::string& op_type,
              const std::string& output_name,
              const std::string& output_arg_name) {
  return GetOp(prog.Block(0), op_type, output_name, output_arg_name);
}

OpDesc* GetOp(const BlockDesc& block_desc,
              const std::string& op_type,
              const std::string& output_name,
              const std::string& output_arg_name) {
  auto all_ops = block_desc.AllOps();
  for (auto* op_desc : all_ops) {
    if (op_desc->Type() == op_type && op_desc->HasOutput(output_name)) {
      const auto& arg_names = op_desc->Outputs().at(output_name);
      for (const auto& name : arg_names) {
        if (name == output_arg_name) return op_desc;
      }
    }
  }
  return nullptr;
}

}  // namespace paddle::framework::ir::test
