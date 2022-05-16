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

#pragma once

#include <functional>
#include <map>

#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/phi/core/ddim.h"

namespace paddle {
namespace framework {
namespace paddle2cinn {

// Class to store the keys for compiling CINN.
//
// CINN cannot handle changable shape now, so CinnCompiler keeps a cache mapping
// from CinnCacheKey to CinnCompiledObject.
//
// The CinnCacheKey contains a graph serialized string and the input tensor
// shapes.
class CinnCacheKey {
 public:
  using GraphHashStrategy = std::function<size_t(const ir::Graph&)>;

  explicit CinnCacheKey(GraphHashStrategy graph_hash);

  CinnCacheKey(const ir::Graph& graph,
               const std::map<std::string, const LoDTensor*>& input_tensors,
               const std::string& arch_str, GraphHashStrategy graph_hash);
  CinnCacheKey(const ir::Graph& graph,
               const std::map<std::string, DDim>& input_shapes,
               const std::string& arch_str, GraphHashStrategy graph_hash);

  ~CinnCacheKey() = default;

  void SetKey(const ir::Graph& graph,
              const std::map<std::string, const LoDTensor*>& input_tensors,
              const std::string& arch_str);
  void SetKey(const ir::Graph& graph,
              const std::map<std::string, DDim>& input_shapes,
              const std::string& arch_str);

  bool operator==(const CinnCacheKey& other) const;
  bool operator!=(const CinnCacheKey& other) const;

  struct Hash {
    static size_t hash_combine(size_t seed, size_t value);
    size_t operator()(const CinnCacheKey& key) const;
  };

 private:
  GraphHashStrategy graph_hash_;
  size_t graph_hash_val_;
  std::map<std::string, DDim> input_shapes_;
  std::string arch_str_;
};

#define CINN_CACHE_KEY_CREATE(NAME)                                    \
  class NAME : public CinnCacheKey {                                   \
   public:                                                             \
    NAME() : CinnCacheKey(HashGraph) {}                                \
                                                                       \
    NAME(const ir::Graph& graph,                                       \
         const std::map<std::string, const LoDTensor*>& input_tensors, \
         const std::string& arch_str)                                  \
        : CinnCacheKey(graph, input_tensors, arch_str, HashGraph) {}   \
                                                                       \
    NAME(const ir::Graph& graph,                                       \
         const std::map<std::string, DDim>& input_shapes,              \
         const std::string& arch_str)                                  \
        : CinnCacheKey(graph, input_shapes, arch_str, HashGraph) {}    \
                                                                       \
   private:                                                            \
    static size_t HashGraph(const ir::Graph& graph);                   \
  };

// Class to store the keys by graph address for compiling CINN.
CINN_CACHE_KEY_CREATE(CinnCacheKeyByAddress)
// Class to store the keys by graph structure for compiling CINN.
CINN_CACHE_KEY_CREATE(CinnCacheKeyByStructure)

#undef CINN_CACHE_KEY_CREATE

}  // namespace paddle2cinn
}  // namespace framework
}  // namespace paddle
