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

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/paddle2cinn/cinn_cache_key.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/platform/macros.h"
#include "paddle/phi/core/utils/rw_lock.h"

namespace cinn {
namespace common {
class Target;
}  // namespace common

namespace hlir::framework {
class GraphCompiler;
class Program;
class Scope;
}  // namespace hlir::framework

namespace auto_schedule {
class AutoTuner;
}  // namespace auto_schedule
}  // namespace cinn

namespace paddle {
namespace operators::details {
class CinnLaunchContext;
}  // namespace operators::details

namespace framework {
namespace paddle2cinn {

struct CinnCompiledObject {
  std::unique_ptr<::cinn::hlir::framework::GraphCompiler> compiler;
  std::unique_ptr<::cinn::auto_schedule::AutoTuner> auto_tuner;
  std::unique_ptr<::cinn::hlir::framework::Program> runtime_program;
  std::shared_ptr<::cinn::hlir::framework::Scope> scope;
  std::unordered_map<std::string, std::string> paddle2cinn_varmap;
  std::unique_ptr<operators::details::CinnLaunchContext> launch_context;
  std::int64_t cached_index;
};

// Entrance to use CINN.
//
// CINN cannot handle changable shape now, so CinnCompiler keeps a cache mapping
// from CinnCacheKey to CinnCompiledObject. If cache hits, we will re-use cache
// stored CinnCompiledObject, otherwise we will compile again and put into
// cache.
class CinnCompiler {
 public:
  // Singleton
  static CinnCompiler* GetInstance();

  const CinnCompiledObject& Compile(
      const ir::Graph& graph,
      const std::map<std::string, const LoDTensor*>& input_tensors,
      const ::cinn::common::Target& target, void* stream = nullptr);

  const CinnCompiledObject& Compile(
      const std::string& compilation_key,
      const std::map<std::string, const LoDTensor*>& input_tensors,
      const ::cinn::common::Target& target, void* stream = nullptr);

  const CinnCompiledObject& GetCompiledObject(int64_t cached_index) const;

  std::string AddGraph(std::unique_ptr<ir::Graph> graph);

  const ir::Graph& FindGraph(const std::string& graph_key) const;

  std::string VizGraph(const std::string& graph_key) const;

  std::string VizGraph(const ir::Graph& graph) const;

  std::string ReadableKey(const std::string& compilation_key) const;

  void Clear();

  std::int64_t real_compiled_num() const { return real_compiled_num_.load(); }

  ~CinnCompiler() = default;

 private:
  CinnCompiler() = default;
  std::unique_ptr<CinnCompiledObject> CompileGraph(
      const ir::Graph& graph,
      const std::map<std::string, const LoDTensor*>& input_tensors,
      const ::cinn::common::Target& target, std::int64_t compiled_num,
      void* stream = nullptr) const;

  // check whether a compiled result is valid by comparing
  // the consistency of external variables of the subgraph
  void CheckCompiledValid(
      const ir::Graph& graph,
      const std::map<std::string, const LoDTensor*>& input_tensors,
      const CinnCompiledObject& compiled_obj) const;

  std::unordered_map<std::string, std::unique_ptr<ir::Graph>> graphs_;
  std::unordered_map<CinnCacheKeyByAddress, std::int64_t, CinnCacheKey::Hash>
      cache_by_address_;
  std::unordered_map<CinnCacheKeyByStructure, std::int64_t, CinnCacheKey::Hash>
      cache_by_struct_;
  std::unordered_map<std::int64_t, std::unique_ptr<CinnCompiledObject>>
      index2cache_;
  std::atomic_int64_t real_compiled_num_{0};
  mutable phi::RWLock rwlock_;

  DISABLE_COPY_AND_ASSIGN(CinnCompiler);
};

}  // namespace paddle2cinn
}  // namespace framework
}  // namespace paddle
