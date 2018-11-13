/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/inference/analysis/pass_manager.h"
#include "paddle/fluid/inference/analysis/fluid_to_data_flow_graph_pass.h"
#include "paddle/fluid/string/pretty_log.h"

namespace paddle {
namespace inference {
namespace analysis {

bool PassManager::Initialize(Argument* argument) {
  argument_ = argument;
  for (auto& pass : data_) {
    VLOG(30) << "Initializing pass [" << pass->repr() << "]";
    if (!pass->Initialize(argument)) {
      LOG(ERROR) << "Failed to initialize pass [" << pass->repr() << "]";
      return false;
    }
  }
  return true;
}

void DfgPassManager::RunAll() {
  PADDLE_ENFORCE(argument_);
  VLOG(30) << "Total " << data_.size() << " Analysys passes";
  for (auto& pass : data_) {
    string::PrettyLogEndl(string::Style::H1(), "* Running Analysis pass [%s]",
                          pass->repr());
    pass->Run(argument_->main_dfg.get());
  }
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
