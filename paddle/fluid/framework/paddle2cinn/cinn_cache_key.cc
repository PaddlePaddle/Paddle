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
#include <sstream>
#include <string>

#include "paddle/common/ddim.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/paddle2cinn/transform_type.h"
#include "paddle/utils/string/string_helper.h"

namespace paddle {
namespace framework {
namespace paddle2cinn {

using GraphHashStrategy = CinnCacheKey::GraphHashStrategy;

CinnCacheKey::CinnCacheKey(GraphHashStrategy graph_hash)
    : graph_hash_(graph_hash) {}

CinnCacheKey::CinnCacheKey(
    const ir::Graph& graph,
    const std::map<std::string, const phi::DenseTensor*>& input_tensors,
    const std::string& arch_str,
    GraphHashStrategy graph_hash)
    : graph_hash_(graph_hash) {
  this->SetKey(graph, input_tensors, arch_str);
}

CinnCacheKey::CinnCacheKey(const ir::Graph& graph,
                           const std::map<std::string, DDim>& input_shapes,
                           const std::map<std::string, DataType>& input_dtypes,
                           const std::string& arch_str,
                           GraphHashStrategy graph_hash)
    : graph_hash_(graph_hash) {
  this->SetKey(graph, input_shapes, input_dtypes, arch_str);
}

void CinnCacheKey::SetKey(
    const ir::Graph& graph,
    const std::map<std::string, const phi::DenseTensor*>& input_tensors,
    const std::string& arch_str) {
  graph_hash_val_ = graph_hash_(graph);
  for (const auto& name_tensor : input_tensors) {
    input_shapes_[name_tensor.first] = name_tensor.second->dims();
    input_dtypes_[name_tensor.first] = name_tensor.second->dtype();
  }
  arch_str_ = arch_str;
}

void CinnCacheKey::SetKey(const ir::Graph& graph,
                          const std::map<std::string, DDim>& input_shapes,
                          const std::map<std::string, DataType>& input_dtypes,
                          const std::string& arch_str) {
  PADDLE_ENFORCE_EQ(
      input_shapes.size(),
      input_dtypes.size(),
      platform::errors::PreconditionNotMet(
          "Required input_shapes has same length with input_dtypes."));

  graph_hash_val_ = graph_hash_(graph);
  input_shapes_ = input_shapes;
  input_dtypes_ = input_dtypes;
  arch_str_ = arch_str;
}

bool CinnCacheKey::operator!=(const CinnCacheKey& other) const {
  return !this->operator==(other);
}

bool CinnCacheKey::operator==(const CinnCacheKey& other) const {
  return graph_hash_val_ == other.graph_hash_val_ &&
         input_shapes_ == other.input_shapes_ &&
         input_dtypes_ == other.input_dtypes_ && arch_str_ == other.arch_str_;
}

size_t CinnCacheKey::Hash::operator()(const CinnCacheKey& key) const {
  std::ostringstream has_str;

  for (const auto& name_shape : key.input_shapes_) {
    has_str << name_shape.first << ",";
    has_str << "[" << name_shape.second << "],";
    PADDLE_ENFORCE_NE(key.input_dtypes_.find(name_shape.first),
                      key.input_dtypes_.end(),
                      platform::errors::PreconditionNotMet(
                          "%s is not in key.input_dtypes_.", name_shape.first));
    has_str << key.input_dtypes_.at(name_shape.first) << ";";
  }

  has_str << key.arch_str_ << ",";
  has_str << key.graph_hash_val_;
  VLOG(1) << "CinnCacheKey : " << has_str.str();
  return std::hash<std::string>()(has_str.str());
}

size_t CinnCacheKeyByStructure::HashGraph(const ir::Graph& graph) {
  // sort grad node by name and id.
  auto compare = [](ir::Node* n1, ir::Node* n2) {
    return (n1->Name() == n2->Name()) ? (n1->id() < n2->id())
                                      : (n1->Name() < n2->Name());
  };

  // graph.Nodes() return unordered_set, here using set to avoid the same graph
  // may return different result
  std::set<ir::Node*, bool (*)(ir::Node*, ir::Node*)> node_set(compare);
  for (ir::Node* node : graph.Nodes()) {
    if (node->IsOp()) {
      // only need cache graph with same op
      node_set.insert(node);
    }
  }

  static const std::unordered_set<std::string> ignore_attr = {
      "op_callstack",
      "op_device",
      "op_namescope",
      "op_role",
      "op_role_var",
      "with_quant_attr"};

  std::set<ir::Node*, bool (*)(ir::Node*, ir::Node*)> input_set(compare),
      output_set(compare);

  std::ostringstream hash_str;
  for (ir::Node* op : node_set) {
    hash_str << op->Name() << ":";
    input_set.clear();
    input_set.insert(op->inputs.begin(), op->inputs.end());
    hash_str << "inputs=["
             << string::join_strings(
                    input_set, ",", [](ir::Node* n) { return n->Name(); })
             << "],";

    output_set.clear();
    output_set.insert(op->outputs.begin(), op->outputs.end());
    hash_str << "outputs=["
             << string::join_strings(
                    output_set, ",", [](ir::Node* n) { return n->Name(); })
             << "],";

    const auto& attrs_unordered_map = op->Op()->GetAttrMap();
    std::map<std::string, Attribute> attrs_map(attrs_unordered_map.begin(),
                                               attrs_unordered_map.end());
    for (const auto& attr : attrs_map) {
      if (ignore_attr.count(attr.first)) {
        continue;
      }
      const auto& attr_str = PaddleAttributeToString(attr.second);
      if (!attr_str.empty()) {
        hash_str << attr.first << "=" << attr_str << ",";
      }
    }
    hash_str << ";";
  }

  VLOG(1) << "The hash graph:\n" << hash_str.str();

  size_t hash_val = std::hash<std::string>()(hash_str.str());
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
