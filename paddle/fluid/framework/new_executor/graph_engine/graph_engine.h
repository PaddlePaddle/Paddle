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

#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/new_executor/interpretercore_util.h"
#include "paddle/fluid/framework/new_executor/new_executor_defs.h"
#include "paddle/fluid/framework/new_executor/profiler.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/phi/backends/device_manager.h"

namespace paddle {
namespace framework {

class GraphEngine {
 public:
  GraphEngine() = default;

  virtual ~GraphEngine() {}

  virtual paddle::framework::interpreter::CostInfo DryRun(
      const std::vector<std::string>& feed_names,
      const std::vector<framework::LoDTensor>& feed_tensors) = 0;

  virtual paddle::framework::FetchList Run(
      const std::vector<std::string>& feed_names) = 0;
};

#ifdef PADDLE_WITH_CUSTOM_DEVICE

class CustomGraphEngine final : public GraphEngine {
 public:
  CustomGraphEngine(const platform::Place& place,
                    Scope* scope,
                    std::shared_ptr<ProgramDesc> prog)
      : place_(place), var_scope_(scope), copy_program_(prog) {
    auto local_scope = &var_scope_.GetMutableScope()->NewScope();
    local_scope_ = local_scope;

    var_scope_.SetLocalScope(local_scope_);
  }

  paddle::framework::interpreter::CostInfo DryRun(
      const std::vector<std::string>& feed_names,
      const std::vector<framework::LoDTensor>& feed_tensors) override {
    paddle::framework::interpreter::CostInfo cost_info;

    return cost_info;
  }

  paddle::framework::FetchList Run(
      const std::vector<std::string>& feed_names) override {
    // phi::DeviceManager::SetDevice(place_);

    const auto& var_list = copy_program_->Block(0).AllVars();

    VLOG(10) << "var_list size = " << var_list.size();
    for (auto& var : var_list) {
      VLOG(10) << "var: " << var->Name();
    }

    paddle::framework::interpreter::build_variable_scope(
        copy_program_->Block(0), &var_scope_, true);

    for (auto& feed_name : feed_names) {
      var_scope_.SetVarSikpInplace(feed_name, true);
    }

    // VLOG(10) << "feed_names size = " << feed_names.size();
    // for (size_t i = 0; i < feed_names.size(); ++i) {
    //   VLOG(10) << "feed var: " << feed_names[i];
    //   auto* feed_var = local_scope_->FindVar(feed_names[i]);
    //   PADDLE_ENFORCE_NOT_NULL(
    //       feed_var,
    //       platform::errors::NotFound("Variable %s should not be nullptr.",
    //                                  feed_names[i]));

    // auto feed_tensor = feed_var->GetMutable<framework::LoDTensor>();
    // feed_tensor->ShareDataWith(feed_tensors[i]);
    // feed_tensor->set_lod(feed_tensors[i].lod());
    // }

    auto* feed_var = local_scope_->FindVar("feed");
    PADDLE_ENFORCE_NOT_NULL(feed_var,
                            platform::errors::NotFound(
                                "Variable %s should not be nullptr.", "feed"));
    auto* feed_list = feed_var->GetMutable<framework::FeedList>();

    VLOG(10) << "feed_list size = " << feed_list->size();
    for (size_t i = 0; i < feed_list->size(); ++i) {
      VLOG(10) << "data = " << paddle::get<0>(feed_list->at(i)).data();
    }

    // return Fetch Tensors
    auto* fetch_var = local_scope_->FindVar(interpreter::kFetchVarName);
    if (fetch_var) {
      return std::move(*fetch_var->GetMutable<framework::FetchList>());
    } else {
      return {};
    }
  }

  bool RunWithFeedAndFetch(const Variable* feed_var, Variable* fetch_var) {
    return true;
  }

 private:
  platform::Place place_;

  VariableScope var_scope_;

  Scope* local_scope_{nullptr};  // not owned

  std::shared_ptr<ProgramDesc> copy_program_{nullptr};
};

#endif

}  // namespace framework
}  // namespace paddle
