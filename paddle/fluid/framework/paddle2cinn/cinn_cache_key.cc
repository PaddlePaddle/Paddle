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

#include <map>
#include <string>

#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/lod_tensor.h"

namespace paddle {
namespace framework {
namespace paddle2cinn {

CinnCacheKey::CinnCacheKey(
    const ir::Graph& graph,
    const std::map<std::string, const LoDTensor*>& feed_tensors) {
  this->SetKey(graph, feed_tensors);
}

CinnCacheKey::CinnCacheKey(const ir::Graph& graph,
                           const std::map<std::string, DDim>& feed_shapes) {
  this->SetKey(graph, feed_shapes);
}

void CinnCacheKey::SetKey(
    const ir::Graph& graph,
    const std::map<std::string, const LoDTensor*>& feed_tensors) {
  ProgramDesc program;
  GraphToProgram(graph, &program);
  program.Proto()->SerializeToString(&graph_serialize_str_);
  for (const auto& name_tensor : feed_tensors) {
    feed_shapes_[name_tensor.first] = name_tensor.second->dims();
  }
}

void CinnCacheKey::SetKey(const ir::Graph& graph,
                          const std::map<std::string, DDim>& feed_shapes) {
  ProgramDesc program;
  GraphToProgram(graph, &program);
  program.Proto()->SerializeToString(&graph_serialize_str_);
  feed_shapes_ = feed_shapes;
}

bool CinnCacheKey::operator!=(const CinnCacheKey& other) const {
  return !this->operator==(other);
}

bool CinnCacheKey::operator==(const CinnCacheKey& other) const {
  return graph_serialize_str_ == other.graph_serialize_str_ &&
         feed_shapes_ == other.feed_shapes_;
}

size_t CinnCacheKey::Hash::hash_combine(size_t seed, size_t value) {
  return seed ^ (value + 0x9e3779b9 + (seed << 6) + (seed >> 2));
}

size_t CinnCacheKey::Hash::operator()(const CinnCacheKey& key) const {
  std::size_t ret = 0;

  std::hash<std::string> string_hasher;
  for (const auto& name_shape : key.feed_shapes_) {
    ret = hash_combine(ret, string_hasher(name_shape.first));
    ret = hash_combine(ret, string_hasher(name_shape.second.to_str()));
  }

  ret = hash_combine(ret, string_hasher(key.graph_serialize_str_));
  return ret;
}

}  // namespace paddle2cinn
}  // namespace framework
}  // namespace paddle
