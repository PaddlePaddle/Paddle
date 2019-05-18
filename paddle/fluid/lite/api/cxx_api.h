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
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "paddle/fluid/lite/core/op_lite.h"
#include "paddle/fluid/lite/core/optimizer.h"
#include "paddle/fluid/lite/core/program.h"
#include "paddle/fluid/lite/core/types.h"
#include "paddle/fluid/lite/model_parser/model_parser.h"

namespace paddle {
namespace lite {

struct Config {};

class ExecutorLite {
 public:
  ExecutorLite() { scope_ = std::make_shared<Scope>(); }
  explicit ExecutorLite(const std::shared_ptr<lite::Scope>& root_scope) {
    scope_ = root_scope;
  }

  void Build(const std::string& model_path, const Place& prefer_place,
             const std::vector<Place>& valid_places) {
    LoadModel(model_path, scope_.get(), &program_desc_);
    Build(program_desc_, prefer_place, valid_places);
  }

  void Build(const framework::proto::ProgramDesc& desc,
             const Place& prefer_place,
             const std::vector<Place>& valid_places) {
    program_desc_ = desc;
    Program program(desc, scope_, valid_places);

    optimizer_.KernelPickPreferPlace(prefer_place);
    core::KernelPickFactor factor;
    factor.ConsiderTarget();
    optimizer_.Run(std::move(program), valid_places, factor);
    program_ = optimizer_.GenRuntimeProgram();
  }

// This method is disabled in mobile, or unnecessary dependencies required.
#ifndef LITE_WITH_LIGHT_WEIGHT_FRAMEWORK
  void SaveModel(const std::string& dir);
#endif

  // Get offset-th col of feed.
  lite::Tensor* GetInput(size_t offset) {
    auto* _feed_list = program_->exec_scope()->FindVar("feed");
    CHECK(_feed_list) << "no feed variable in exec_scope";
    auto* feed_list = _feed_list->GetMutable<std::vector<lite::Tensor>>();
    if (offset >= feed_list->size()) {
      feed_list->resize(offset + 1);
    }
    return &feed_list->at(offset);
  }

  const lite::Tensor* GetOutput(size_t offset) {
    auto* _fetch_list = program_->exec_scope()->FindVar("fetch");
    CHECK(_fetch_list) << "no fatch variable in exec_scope";
    auto& fetch_list = *_fetch_list->GetMutable<std::vector<lite::Tensor>>();
    CHECK_LT(offset, fetch_list.size()) << "offset " << offset << " overflow";
    return &fetch_list.at(offset);
  }

  void Run() { program_->Run(); }

  const framework::proto::ProgramDesc& program_desc() const {
    return program_desc_;
  }

 private:
  Optimizer optimizer_;
  framework::proto::ProgramDesc program_desc_;
  std::shared_ptr<Scope> scope_;
  std::unique_ptr<RuntimeProgram> program_;
};

/*
 * An executor for training.
 *
 * Usage:
 *
 * CXXTrainer trainer(...);
 * trainer.RunStartupProgram(...);
 * auto exe = BuildMainProgramExecutor(...);
 *
 * for (auto& epoch : epoches) {
 *   auto* tensor0 = exe.GetInput(...);
 *   // fill data for tensor0
 *   exe.Run();
 * }
 */
class CXXTrainer {
 public:
  CXXTrainer(const std::shared_ptr<lite::Scope>& root_scope,
             const Place& preferred_place,
             const std::vector<Place>& valid_places)
      : scope_(root_scope),
        preferred_place_(preferred_place),
        valid_places_(valid_places),
        main_program_executor_(ExecutorLite(scope_)) {}

  // Build the RuntimeProgram cache for the main program. The cache will run
  // multiple times for the epoches.
  // NOTE Just support to execute the 0-th block currently.
  ExecutorLite& BuildMainProgramExecutor(
      const framework::proto::ProgramDesc& desc, int block_id = 0) {
    main_program_executor_.Build(desc, preferred_place_, valid_places_);
    return main_program_executor_;
  }

  // Run the startup program. It just executes once, no cache needed.
  void RunStartupProgram(const framework::proto::ProgramDesc& desc,
                         int block_id = 0) {
    ExecutorLite exe(scope_);
    exe.Build(desc, preferred_place_, valid_places_);
    exe.Run();
  }

 private:
  std::shared_ptr<lite::Scope> scope_;

  Place preferred_place_;
  std::vector<Place> valid_places_;

  // The training program.
  ExecutorLite main_program_executor_;
};

}  // namespace lite
}  // namespace paddle
