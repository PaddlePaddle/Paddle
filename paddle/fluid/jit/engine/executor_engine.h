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

#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/scope.h"

#include "paddle/fluid/jit/engine/base_engine.h"
#include "paddle/fluid/jit/function_schema.h"
#include "paddle/fluid/jit/function_utils.h"

namespace paddle {
namespace jit {

class ExecutorEngine : public BaseEngine {
 public:
  ExecutorEngine(const std::shared_ptr<FunctionInfo> &info,
                 const VariableMap &params_dict,
                 const phi::Place &place);

  ~ExecutorEngine() noexcept {}

  std::vector<Tensor> operator()(const std::vector<Tensor> &inputs);

  std::vector<DenseTensor> operator()(const std::vector<DenseTensor> &inputs);

  const std::shared_ptr<FunctionInfo> &Info() const;

 private:
  std::shared_ptr<FunctionInfo> info_;
  framework::Scope scope_;
  phi::Place place_;
  framework::Executor inner_exe_;
};

}  // namespace jit
}  // namespace paddle
