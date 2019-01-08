// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
#include <vector>
#include "paddle/fluid/framework/details/op_handle_base.h"
#include "paddle/fluid/framework/details/var_handle.h"

#include "paddle/fluid/framework/details/execution_strategy.h"
#include "paddle/fluid/framework/details/ssa_graph_executor.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/platform/place.h"
namespace paddle {
namespace framework {
namespace details {

struct VariableInfo {
  std::string name_;
  proto::VarType::Type type_;
  bool persistable_;
};

class ScopeBufferedSSAGraphExecutor : public SSAGraphExecutor {
 public:
  ScopeBufferedSSAGraphExecutor(
      ExecutionStrategy strategy, std::vector<Scope*> local_scopes,
      std::vector<VariableInfo> var_infos, std::vector<platform::Place> places,
      std::unique_ptr<SSAGraphExecutor>&& underlying_executor);

  const ir::Graph& Graph() const override {
    return underlying_executor_->Graph();
  }

  FeedFetchList Run(const std::vector<std::string>& fetch_tensors) override;

 private:
  inline void WaitComputationalStreams() {
    // Wait All computational streams
    for (auto p : places_) {
      platform::DeviceContextPool::Instance().Get(p)->Wait();
    }
  }

 private:
  size_t drop_scope_counter_{0};

  ExecutionStrategy strategy_;
  std::unique_ptr<SSAGraphExecutor> underlying_executor_;
  std::vector<Scope*> local_scopes_;
  std::vector<VariableInfo> var_infos_;
  std::vector<platform::Place> places_;
};
}  // namespace details
}  // namespace framework
}  // namespace paddle
