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
#include <utility>
#include <vector>

#include "ThreadPool.h"
#include "paddle/fluid/framework/details/threaded_ssa_graph_executor.h"

namespace paddle {
namespace framework {
namespace details {

struct VarInfo {
  std::string name_;
  proto::VarType::Type type_;
  bool persistable_;
};

class AsyncSSAGraphExecutor : public SSAGraphExecutor {
 public:
  AsyncSSAGraphExecutor(const ExecutionStrategy &strategy,
                        const std::vector<Scope *> &local_scopes,
                        const std::vector<Scope *> &local_exec_scopes,
                        const std::vector<platform::Place> &places,
                        std::vector<ir::Graph *> graphs);
  ~AsyncSSAGraphExecutor() final = default;
  const ir::Graph &Graph() const override { return *graphs_[0]; }

  FetchResultType Run(const std::vector<std::string> &fetch_tensors,
                      bool return_merged) override;

 private:
  void StartOffPythonTrainLoop(bool return_merged);
  void HandleException();

 private:
  ExecutionStrategy strategy_;
  std::vector<Scope *> local_scopes_;
  std::vector<Scope *> local_exec_scopes_;
  std::unique_ptr<::ThreadPool> pool_{nullptr};
  std::vector<platform::Place> places_;
  std::vector<ir::Graph *> graphs_;

  std::vector<std::unique_ptr<details::ThreadedSSAGraphExecutor>> executors_;
  ExceptionHolder exception_holder_;
  std::vector<std::future<void>> run_futures_;
  std::vector<VarInfo> var_infos_;
};

}  // namespace details
}  // namespace framework
}  // namespace paddle
