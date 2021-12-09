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

#include "paddle/fluid/framework/paddle2cinn/cinn_cache_key.h"

#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <string>

#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/inference/analysis/dot.h"

namespace paddle {
namespace framework {
namespace paddle2cinn {

CinnCacheKey::CinnCacheKey(
    const ir::Graph& graph,
    const std::map<std::string, const LoDTensor*>& input_tensors,
    const std::string& arch_str) {
  this->SetKey(graph, input_tensors, arch_str);
}

CinnCacheKey::CinnCacheKey(const ir::Graph& graph,
                           const std::map<std::string, DDim>& input_shapes,
                           const std::string& arch_str) {
  this->SetKey(graph, input_shapes, arch_str);
}

size_t CinnCacheKey::HashGraph(const ir::Graph& graph) {
  // using Dot to unqiue graph
  inference::analysis::Dot dot;
  std::unordered_map<const ir::Node*, std::string> node2dot;
  int id = 0;
  // Create nodes
  // graph.Nodes() return unordered_set, the same graph may
  // return different result?
  for (const ir::Node* n : graph.Nodes()) {
    std::string node_id = std::to_string(id++);
    dot.AddNode(node_id, {}, n->Name(), true);
    node2dot[n] = node_id;
  }

  // Create edges
  for (const ir::Node* n : graph.Nodes()) {
    const auto& src_id = node2dot.at(n);
    for (auto* out : n->outputs) {
      const auto& dest_id = node2dot.at(out);
      dot.AddEdge(src_id, dest_id, {});
    }
  }

  const std::string& viz_graph = dot.Build();
  VLOG(1) << "The hash graph:\n" << viz_graph;

  size_t hash_val = std::hash<std::string>()(viz_graph);
  VLOG(4) << "The graph's hash value is: " << hash_val;
  return hash_val;
}

void CinnCacheKey::SetKey(
    const ir::Graph& graph,
    const std::map<std::string, const LoDTensor*>& input_tensors,
    const std::string& arch_str) {
  graph_serialize_str_ = std::to_string(HashGraph(graph));
  for (const auto& name_tensor : input_tensors) {
    input_shapes_[name_tensor.first] = name_tensor.second->dims();
  }
  arch_str_ = arch_str;
}

void CinnCacheKey::SetKey(const ir::Graph& graph,
                          const std::map<std::string, DDim>& input_shapes,
                          const std::string& arch_str) {
  graph_serialize_str_ = std::to_string(HashGraph(graph));
  input_shapes_ = input_shapes;
  arch_str_ = arch_str;
}

bool CinnCacheKey::operator!=(const CinnCacheKey& other) const {
  return !this->operator==(other);
}

bool CinnCacheKey::operator==(const CinnCacheKey& other) const {
  return graph_serialize_str_ == other.graph_serialize_str_ &&
         input_shapes_ == other.input_shapes_ && arch_str_ == other.arch_str_;
}

size_t CinnCacheKey::Hash::hash_combine(size_t seed, size_t value) {
  return seed ^ (value + 0x9e3779b9 + (seed << 6) + (seed >> 2));
}

size_t CinnCacheKey::Hash::operator()(const CinnCacheKey& key) const {
  std::size_t ret = 0;

  std::hash<std::string> string_hasher;
  for (const auto& name_shape : key.input_shapes_) {
    ret = hash_combine(ret, string_hasher(name_shape.first));
    ret = hash_combine(ret, string_hasher(name_shape.second.to_str()));
  }

  ret = hash_combine(ret, string_hasher(key.graph_serialize_str_));
  ret = hash_combine(ret, string_hasher(key.arch_str_));
  return ret;
}

}  // namespace paddle2cinn
}  // namespace framework
}  // namespace paddle
