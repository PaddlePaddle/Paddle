// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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
#include <absl/container/flat_hash_map.h>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "paddle/cinn/frontend/syntax.h"
#include "paddle/cinn/hlir/framework/graph_compiler.h"
#include "paddle/cinn/hlir/framework/scope.h"

namespace cinn {
namespace frontend {

/**
 * The executor for a model.
 */
class Interpreter final {
 public:
  Interpreter(const std::vector<std::string>& input_names,
              const std::vector<hlir::framework::shape_t>& input_shapes);

  /**
   * Load a Paddle model.
   * @param model_dir The directory path to the model.
   * @param params_combined Whether the parameters are composed to a single
   * file.
   */
  void LoadPaddleModel(const std::string& model_dir,
                       const Target& target,
                       bool params_combined,
                       const std::string& model_name = "");

  /**
   * Run the executor.
   */
  void Run();

  frontend::Program GetProgram();

  hlir::framework::Tensor GetTensor(const std::string& name);

  std::shared_ptr<hlir::framework::Scope> GetScope();

  ~Interpreter();

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace frontend
}  // namespace cinn
