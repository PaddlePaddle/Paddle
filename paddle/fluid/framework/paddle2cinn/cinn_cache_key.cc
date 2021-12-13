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

namespace paddle {
namespace framework {
namespace paddle2cinn {

using GraphHashStrategy = CinnCacheKey::GraphHashStrategy;

CinnCacheKey::CinnCacheKey(GraphHashStrategy graph_hash)
    : graph_hash_(graph_hash) {}

CinnCacheKey::CinnCacheKey(
    const ir::Graph& graph,
    const std::map<std::string, const LoDTensor*>& input_tensors,
    const std::string& arch_str, GraphHashStrategy graph_hash)
    : graph_hash_(graph_hash) {
  this->SetKey(graph, input_tensors, arch_str);
}

CinnCacheKey::CinnCacheKey(const ir::Graph& graph,
                           const std::map<std::string, DDim>& input_shapes,
                           const std::string& arch_str,
                           GraphHashStrategy graph_hash)
    : graph_hash_(graph_hash) {
  this->SetKey(graph, input_shapes, arch_str);
}

void CinnCacheKey::SetKey(
    const ir::Graph& graph,
    const std::map<std::string, const LoDTensor*>& input_tensors,
    const std::string& arch_str) {
  graph_hash_val_ = graph_hash_(graph);
  for (const auto& name_tensor : input_tensors) {
    input_shapes_[name_tensor.first] = name_tensor.second->dims();
  }
  arch_str_ = arch_str;
}

void CinnCacheKey::SetKey(const ir::Graph& graph,
                          const std::map<std::string, DDim>& input_shapes,
                          const std::string& arch_str) {
  graph_hash_val_ = graph_hash_(graph);
  input_shapes_ = input_shapes;
  arch_str_ = arch_str;
}

bool CinnCacheKey::operator!=(const CinnCacheKey& other) const {
  return !this->operator==(other);
}

bool CinnCacheKey::operator==(const CinnCacheKey& other) const {
  return graph_hash_val_ == other.graph_hash_val_ &&
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

  ret = hash_combine(ret, key.graph_hash_val_);
  ret = hash_combine(ret, string_hasher(key.arch_str_));
  return ret;
}

size_t CinnCacheKeyByStructure::HashGraph(const ir::Graph& graph) {
  // sort grad node by name and id.
  auto compare = [](ir::Node* n1, ir::Node* n2) {
    return (n1->Name() == n2->Name()) ? (n1->id() < n2->id())
                                      : (n1->Name() < n2->Name());
  };

  // graph.Nodes() return unordered_set, here using set to avoid the same graph
  // may return different result
  std::set<ir::Node *, bool (*)(ir::Node *, ir::Node *)> node_set(compare),
      output_set(compare);
  node_set.insert(graph.Nodes().begin(), graph.Nodes().end());

  std::string hash_str;
  for (ir::Node* n : node_set) {
    hash_str.append(n->Name());

    output_set.clear();
    output_set.insert(n->outputs.begin(), n->outputs.end());
    for (auto* out : output_set) {
      hash_str.append(out->Name());
    }
  }

  VLOG(1) << "The hash graph:\n" << hash_str;

  size_t hash_val = std::hash<std::string>()(hash_str);
  VLOG(4) << "The graph's hash value by graph structure is: " << hash_val;
  return hash_val;
}

size_t CinnCacheKeyByAddress::HashGraph(const ir::Graph& graph) {
  size_t hash_val = reinterpret_cast<size_t>(&graph);
  VLOG(4) << "The graph's hash value by graph address is: " << hash_val;
  return hash_val;
}

}  // namespace paddle2cinn
}  // namespace framework
}  // namespace paddle
