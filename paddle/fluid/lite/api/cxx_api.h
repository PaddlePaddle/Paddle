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
#include "paddle/fluid/lite/core/op_executor.h"
#include "paddle/fluid/lite/core/op_lite.h"
#include "paddle/fluid/lite/core/optimizer.h"
#include "paddle/fluid/lite/core/program.h"
#include "paddle/fluid/lite/core/types.h"
#include "paddle/fluid/lite/model_parser/model_parser.h"

namespace paddle {
namespace lite {

struct Config {};

class Predictor {
 public:
  Predictor() { scope_ = std::make_shared<Scope>(); }

  void Build(const std::string& model_path,
             const std::vector<Place>& valid_places) {
    framework::proto::ProgramDesc prog;
    LoadModel(model_path, scope_.get(), &prog);
    framework::ProgramDesc prog_desc(prog);

    Program program(prog_desc, scope_, valid_places);

    Optimizer optimizer;
    core::KernelPickFactor factor;
    factor.ConsiderTarget();
    optimizer.Run(std::move(program), valid_places, factor);
    program_ = optimizer.GenRuntimeProgram();
  }

  // Get offset-th col of feed.
  Tensor* GetInput(size_t offset) {
    auto* _feed_list = program_->exec_scope()->FindVar("feed");
    CHECK(_feed_list) << "no feed variable in exec_scope";
    auto* feed_list = _feed_list->GetMutable<std::vector<Tensor>>();
    if (offset >= feed_list->size()) {
      feed_list->resize(offset + 1);
    }
    return &feed_list->at(offset);
  }

  const Tensor* GetOutput(size_t offset) {
    auto* _fetch_list = program_->exec_scope()->FindVar("fetch");
    CHECK(_fetch_list) << "no fatch variable in exec_scope";
    auto& fetch_list = *_fetch_list->GetMutable<std::vector<lite::Tensor>>();
    CHECK_LT(offset, fetch_list.size()) << "offset " << offset << " overflow";
    return &fetch_list.at(offset);
  }

  void Run() { program_->Run(); }

 private:
  std::shared_ptr<Scope> scope_;
  std::unique_ptr<RuntimeProgram> program_;
};

}  // namespace lite
}  // namespace paddle
