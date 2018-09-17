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

#include "paddle/fluid/framework/details/build_strategy.h"

#include <string>
#include <tuple>

#include "paddle/fluid/framework/details/multi_devices_graph_check_pass.h"
#include "paddle/fluid/framework/details/multi_devices_graph_print_pass.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_viz_pass.h"

namespace paddle {
namespace framework {
namespace details {

class ParallelExecutorPassBuilder : public ir::PassBuilder {
 public:
  explicit ParallelExecutorPassBuilder(const BuildStrategy &strategy)
      : ir::PassBuilder(), strategy_(strategy) {
    // Apply a graph viz pass to record a graph.
    if (!strategy_.debug_graphviz_path_.empty()) {
      auto viz_pass = AppendPass("graph_viz_pass");
      const std::string graph_path = string::Sprintf(
          "%s%s", strategy_.debug_graphviz_path_.c_str(), "_original_graph");
      viz_pass->Set<std::string>("graph_viz_path", new std::string(graph_path));
    }

    // Apply op fusion.
    if (strategy.fuse_elewise_add_act_ops_) {
      auto fuse_elewise_add_act_pass =
          ir::PassRegistry::Instance().Get("fuse_elewise_add_act_pass");
      graph = fuse_elewise_add_act_pass->Apply(std::move(graph));
      // Apply a graph viz pass to record a graph.
      if (!strategy.debug_graphviz_path_.empty()) {
        auto viz_pass = ir::PassRegistry::Instance().Get("graph_viz_pass");
        const std::string graph_path = string::Sprintf(
            "%s%s", strategy.debug_graphviz_path_.c_str(), "_fused_graph");
        viz_pass->Set<std::string>("graph_viz_path", new std::string(graph_path));
        graph = viz_pass->Apply(std::move(graph));
      }
    }

    // Convert graph to run on multi-devices.
    auto multi_devices_pass = AppendPass("multi_devices_pass");
    multi_devices_pass->SetNotOwned<const BuildStrategy>("strategy",
                                                         &strategy_);

    // Apply a graph print pass to record a graph with device info.
    if (!strategy_.debug_graphviz_path_.empty()) {
      auto multi_devices_print_pass = AppendPass("multi_devices_print_pass");
      multi_devices_print_pass->SetNotOwned<const std::string>(
          "debug_graphviz_path", &strategy_.debug_graphviz_path_);
      multi_devices_print_pass->Set<details::GraphvizSSAGraphPrinter>(
          "graph_printer", new details::GraphvizSSAGraphPrinter);
    }

    // Verify that the graph is correct for multi-device executor.
    AppendPass("multi_devices_check_pass");
  }

  std::unique_ptr<ir::Graph> Build(
      const ProgramDesc &main_program,
      const std::vector<platform::Place> &places,
      const std::string &loss_var_name,
      const std::unordered_set<std::string> &param_names,
      const std::vector<Scope *> &local_scopes,
#ifdef PADDLE_WITH_CUDA
      const bool use_cuda, platform::NCCLContextMap *nccl_ctxs) const {
#else
      const bool use_cuda) const {
#endif
    // Convert the program to graph.
    std::unique_ptr<ir::Graph> graph(new ir::Graph(main_program));

    for (std::shared_ptr<ir::Pass> &pass : AllPasses()) {
      if (pass->Type() == "multi_devices_pass") {
        pass->SetNotOwned<const std::vector<platform::Place>>("places",
                                                              &places);
        pass->SetNotOwned<const std::string>("loss_var_name", &loss_var_name);
        pass->SetNotOwned<const std::unordered_set<std::string>>("params",
                                                                 &param_names);
        pass->SetNotOwned<const std::vector<Scope *>>("local_scopes",
                                                      &local_scopes);
#ifdef PADDLE_WITH_CUDA
        platform::NCCLContextMap *nctx = use_cuda ? nccl_ctxs : nullptr;
        pass->SetNotOwned<platform::NCCLContextMap>("nccl_ctxs", nctx);
#endif
      }
      graph = pass->Apply(std::move(graph));
    }
    return graph;
  }

 private:
  BuildStrategy strategy_;
};

ir::PassBuilder *BuildStrategy::CreatePassBuilder() const {
  pass_builder_.reset(new ParallelExecutorPassBuilder(*this));
  return pass_builder_.get();
}

std::unique_ptr<ir::Graph> BuildStrategy::Apply(
    const ProgramDesc &main_program, const std::vector<platform::Place> &places,
    const std::string &loss_var_name,
    const std::unordered_set<std::string> &param_names,
    const std::vector<Scope *> &local_scopes,
#ifdef PADDLE_WITH_CUDA
    const bool use_cuda, platform::NCCLContextMap *nccl_ctxs) const {
#else
    const bool use_cuda) const {
#endif
  if (!pass_builder_) {
    CreatePassBuilder();
  }
  // std::unique_ptr<ir::Graph> graph;
  ParallelExecutorPassBuilder *builder =
      reinterpret_cast<ParallelExecutorPassBuilder *>(pass_builder_.get());
#ifdef PADDLE_WITH_CUDA
  std::unique_ptr<ir::Graph> graph =
      builder->Build(main_program, places, loss_var_name, param_names,
                     local_scopes, use_cuda, nccl_ctxs);
#else
  std::unique_ptr<ir::Graph> graph = builder->Build(
      main_program, places, loss_var_name, param_names, local_scopes, use_cuda);
#endif
  return graph;
}
}  // namespace details
}  // namespace framework
}  // namespace paddle

USE_PASS(fuse_elewise_add_act_pass);
USE_PASS(graph_viz_pass);
USE_PASS(multi_devices_pass);
USE_PASS(multi_devices_check_pass);
USE_PASS(multi_devices_print_pass);
