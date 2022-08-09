// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <vector>

#include "paddle/fluid/framework/scope.h"

#include "paddle/fluid/jit/engine/base_engine.h"
#include "paddle/fluid/jit/function_schema.h"
#include "paddle/fluid/jit/function_utils.h"

namespace paddle {

namespace framework {
class ParallelExecutor;
namespace details {
class ExecutionStrategy;
}
namespace ir {
class Graph;
}
}  // namespace framework

namespace jit {
using ExecutionStrategy = framework::details::ExecutionStrategy;
using ParallelExecutor = framework::ParallelExecutor;
using Graph = framework::ir::Graph;

class PEEngine : public BaseEngine {
 public:
  PEEngine(const std::shared_ptr<FunctionInfo> &info,
           const VariableMap &params_dict,
           const phi::Place &place);

  ~PEEngine() noexcept {}

  void CreateGraphAndPE();

  std::vector<Tensor> operator()(const std::vector<Tensor> &inputs);

  std::vector<DenseTensor> operator()(const std::vector<DenseTensor> &inputs);

  const std::shared_ptr<FunctionInfo> &Info() const;

 private:
  std::shared_ptr<FunctionInfo> info_;
  framework::Scope scope_;
  phi::Place place_;
  std::shared_ptr<ParallelExecutor> inner_pe_;
  std::shared_ptr<Graph> graph_;
};

}  // namespace jit
}  // namespace paddle
