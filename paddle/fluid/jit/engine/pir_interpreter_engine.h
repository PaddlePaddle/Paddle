// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/pir/include/core/program.h"

#include "paddle/fluid/framework/new_executor/pir_interpreter.h"
#include "paddle/fluid/jit/engine/base_engine.h"
#include "paddle/fluid/jit/function_schema.h"
#include "paddle/fluid/jit/function_utils.h"

namespace paddle {

namespace framework {
class InterpreterCore;
}  // namespace framework

namespace jit {
using InterpreterCore = framework::InterpreterCore;
// using Graph = framework::ir::Graph;
using PirInterpreter = framework::PirInterpreter;

class PirInterpreterEngine : public BaseEngine {
 public:
  PirInterpreterEngine(const std::shared_ptr<PirFunctionInfo> &info,
                       const std::shared_ptr<VariableMap> &params_dict,
                       const phi::Place &place,
                       const std::shared_ptr<pir::Program> &prog);

  ~PirInterpreterEngine() noexcept {}

  void CreateInterpreterCore();

  std::vector<Tensor> operator()(const std::vector<Tensor> &inputs) override;

  std::vector<DenseTensor> operator()(
      const std::vector<DenseTensor> &inputs) override;

  const std::shared_ptr<PirFunctionInfo> &Info() const;

  std::unique_ptr<BaseEngine> Clone(void *stream = nullptr) override;

 private:
  std::shared_ptr<PirFunctionInfo> info_;
  std::shared_ptr<VariableMap> params_dict_;
  framework::Scope scope_;
  phi::Place place_;
  std::shared_ptr<framework::PirInterpreter> inner_interpreter_;
  std::shared_ptr<pir::Program> prog_;
};

}  // namespace jit
}  // namespace paddle
