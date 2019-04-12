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

#pragma once
#include "paddle/fluid/lite/core/executor.h"
#include "paddle/fluid/lite/core/op_lite.h"
#include "paddle/fluid/lite/model_parser/model_parser.h"

namespace paddle {
namespace lite {

struct Config {};

class Predictor {
 public:
  void Build(const std::string& model_path,
             const std::vector<OpLite::Place>& valid_places) {
    CHECK(!executor_.get()) << "duplicate build found";
    framework::proto::ProgramDesc prog;
    LoadModel(model_path, &scope_, &prog);
    framework::ProgramDesc prog_desc(prog);

    executor_.reset(new Executor(&scope_, valid_places));
    executor_->PrepareWorkspace(prog_desc);
    executor_->Build(prog_desc);
  }

  // Get a tensor for input from scope directly.
  Tensor* GetInputTensor(const std::string& name) {
    auto* var = executor_->exec_scope()->FindVar(name);
    CHECK(var) << "no tensor called " << name << " exists";
    return var->GetMutable<Tensor>();
  }

  // Get a tensor for output from scope directly.
  const Tensor* GetOutputTensor(const std::string& name) {
    auto* var = executor_->exec_scope()->FindVar(name);
    CHECK(var) << "no tensor called " << name << " exists";
    return &var->Get<Tensor>();
  }

  void Run() { executor_->Run(); }

 private:
  Scope scope_;
  std::unique_ptr<lite::Executor> executor_;
};

}  // namespace lite
}  // namespace paddle
