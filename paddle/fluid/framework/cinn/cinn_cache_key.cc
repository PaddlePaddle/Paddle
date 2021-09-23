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

#include "paddle/fluid/framework/cinn/cinn_cache_key.h"

#include <map>
#include <string>

#include <boost/functional/hash.hpp>

#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/lod_tensor.h"

namespace paddle {
namespace framework {
namespace cinn {

CinnCacheKey::CinnCacheKey(
    const ir::Graph* graph,
    const std::map<std::string, const LoDTensor*>& feed_tensors) {
  this->SetKey(graph, feed_tensors);
}

CinnCacheKey::CinnCacheKey(const ir::Graph* graph,
                           const std::map<std::string, DDim>& feed_shapes) {
  this->SetKey(graph, feed_shapes);
}

void CinnCacheKey::SetKey(
    const ir::Graph* graph,
    const std::map<std::string, const LoDTensor*>& feed_tensors) {
  graph_ = graph;
  for (const auto& name_tensor : feed_tensors) {
    feed_shapes_[name_tensor.first] = name_tensor.second->dims();
  }
}

void CinnCacheKey::SetKey(const ir::Graph* graph,
                          const std::map<std::string, DDim>& feed_shapes) {
  graph_ = graph;
  feed_shapes_ = feed_shapes;
}

bool CinnCacheKey::operator!=(const CinnCacheKey& other) const {
  return !this->operator==(other);
}

bool CinnCacheKey::operator==(const CinnCacheKey& other) const {
  if (feed_shapes_ != other.feed_shapes_) {
    return false;
  }
  if ((graph_ == nullptr && other.graph_ != nullptr) ||
      (graph_ != nullptr && other.graph_ == nullptr)) {
    return false;
  }
  if (graph_ == nullptr && other.graph_ == nullptr) {
    return true;
  }

  // graph_ and other.graph_ are not NULL and shapes are equal
  ProgramDesc program;
  ProgramDesc other_program;
  GraphToProgram(*graph_, &program);
  GraphToProgram(*(other.graph_), &other_program);
  return program.Proto()->SerializeAsString() ==
         other_program.Proto()->SerializeAsString();
}

size_t CinnCacheKey::Hash::operator()(const CinnCacheKey& key) const {
  std::size_t ret = 0;

  std::hash<std::string> string_hasher;
  for (const auto& name_shape : key.feed_shapes_) {
    boost::hash_combine(ret, string_hasher(name_shape.first));
    boost::hash_combine(ret, string_hasher(name_shape.second.to_str()));
  }

  if (key.graph_ != nullptr) {
    ProgramDesc program;
    GraphToProgram(*key.graph_, &program);
    std::string prog_serialize_str = program.Proto()->SerializeAsString();
    boost::hash_combine(ret, string_hasher(prog_serialize_str));
  }

  return ret;
}

}  // namespace cinn
}  // namespace framework
}  // namespace paddle
