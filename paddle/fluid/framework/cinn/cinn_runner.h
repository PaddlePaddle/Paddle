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

#include <map>
#include <memory>
#include <unordered_map>

#include "paddle/fluid/framework/cinn/cinn_cache_key.h"
#include "paddle/fluid/framework/cinn/cinn_compiled_object.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/scope.h"

namespace paddle {
namespace framework {
namespace cinn {

// Entrance to run CINN
class CinnRunner {
 public:
  CinnRunner() {}
  ~CinnRunner() {}

  // Feed LoDTensors to tun CINN compiled object and return fetched result
  std::map<std::string, FetchType*> Run(
      const ir::Graph& graph, Scope* scope,
      std::map<std::string, const LoDTensor*>* feed_targets);

 private:
  std::unordered_map<CinnCacheKey, std::shared_ptr<CinnCompiledObject>,
                     CinnCacheKey::Hash>
      cache_;
};

}  // namespace cinn
}  // namespace framework
}  // namespace paddle
