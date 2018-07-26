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

#include "paddle/fluid/framework/details/ssa_graph_builder_factory.h"
#include <fstream>
#include "paddle/fluid/framework/details/multi_devices_graph_builder.h"
#include "paddle/fluid/framework/details/ssa_graph_checker.h"
#include "paddle/fluid/framework/details/ssa_graph_printer.h"

namespace paddle {
namespace framework {
namespace details {
std::unique_ptr<ir::Pass> ParallelExecutorPassManager::Create() {
  std::unique_ptr<ir::Pass> res(new MultiDevSSAGraphBuilder);
  res->SetNotOwned<std::vector<platform::Place>>("places", &places_);
  res->SetNotOwned<std::string>("loss_var_name", &loss_var_name_);
  res->SetNotOwned<std::unordered_set<std::string>>("params", &param_names_);
  res->SetNotOwned<std::vector<Scope *>>("local_scopes", &local_scopes_);
  res->SetNotOwned<BuildStrategy>("strategy", &strategy_);
#ifdef PADDLE_WITH_CUDA
  res->SetNotOwned<platform::NCCLContextMap>("nccl_ctxs", nccl_ctxs_);
#endif

  if (!strategy_.debug_graphviz_path_.empty()) {
    ir::Pass *previous_pass = res.release();
    res.reset(new SSAGraghBuilderWithPrinter);
    res->Set<ir::Pass>("previous_pass", previous_pass);
    res->SetNotOwned<std::string>("debug_graphviz_path",
                                  &strategy_.debug_graphviz_path_);
    res->Set<GraphvizSSAGraphPrinter>("graph_printer",
                                      new GraphvizSSAGraphPrinter);
  }

  ir::Pass *previous_pass = res.release();
  res.reset(new SSAGraghBuilderWithChecker);
  res->Set<ir::Pass>("previous_pass", previous_pass);

  return res;
}
}  // namespace details
}  // namespace framework
}  // namespace paddle
