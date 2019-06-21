// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

/*
 * This file implements a light-weight API which can run on mobile. We limit the
 * dependencies and the runtime computation complexity.
 */
#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "paddle/fluid/lite/core/compatible_tensor.h"
#include "paddle/fluid/lite/core/context.h"
#include "paddle/fluid/lite/core/program.h"
#include "paddle/fluid/lite/core/types.h"
#include "paddle/fluid/lite/model_parser/model_parser.h"
#include "paddle/fluid/lite/model_parser/pb/op_desc.h"

namespace paddle {
namespace lite {

/*
 * The light weight predictor, mainly for mobile. It loads an optimized model,
 * and will not depend on the MIR or perform latter optimization.
 */
class LightPredictor {
 public:
  explicit LightPredictor(const std::string& model_dir);

  void Run() { program_->Run(); }

  // Get offset-th col of feed inputs.
  Tensor* GetInput(size_t offset);

  // Get offset-th col of fetch outputs.
  const Tensor* GetOutput(size_t offset);

  const lite::Tensor* GetTensor(const std::string& name) const {
    auto* var = program_->exec_scope()->FindVar(name);
    return &var->Get<lite::Tensor>();
  }

 private:
  void Build(const std::string& model_dir);
  void BuildRuntimeProgram(const framework::proto::ProgramDesc& prog);

 private:
  std::shared_ptr<Scope> scope_;
  std::unique_ptr<RuntimeProgram> program_;
};

}  // namespace lite
}  // namespace paddle
