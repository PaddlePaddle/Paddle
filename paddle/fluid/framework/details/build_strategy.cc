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

#include <glog/logging.h>
#include <memory>

#include "paddle/fluid/framework/details/cfg_graph.h"
#include "paddle/fluid/framework/details/multi_devices_graph_check_pass.h"
#include "paddle/fluid/framework/details/multi_devices_graph_print_pass.h"
#include "paddle/fluid/framework/details/sequential_execution_pass.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/graph_viz_pass.h"

namespace paddle {
namespace framework {
namespace details {

class ParallelExecutorPassBuilder : public ir::PassBuilder {
 public:
  explicit ParallelExecutorPassBuilder(const BuildStrategy &strategy)
      : ir::PassBuilder(), strategy_(strategy) {
    // NOTE(dzh): memory optimize should be a runtime pass.
    // However, after multi_devices_pass, VarHandle, OpHandle is
    // the de-fact IR, any reuse on Graph is meaningless.
    // A side-effect of that, memory optimize cannot forsee the fetched vars
    // , so fetchlist should be set persistable before call the Run interface.
    if (strategy.memory_optimize_) {
      auto analysis_var_pass = AppendPass("analysis_var_pass");
    }

    if (strategy_.enable_sequential_execution_) {
      AppendPass("sequential_execution_pass");
    }

    // Add a graph viz pass to record a graph.
    if (!strategy_.debug_graphviz_path_.empty()) {
      auto viz_pass = AppendPass("graph_viz_pass");
      const std::string graph_path = string::Sprintf(
          "%s%s", strategy_.debug_graphviz_path_.c_str(), "_original_graph");
      viz_pass->Set<std::string>("graph_viz_path", new std::string(graph_path));
    }

    // Add op fusion.
    if (strategy.fuse_elewise_add_act_ops_) {
      auto fuse_elewise_add_act_pass = AppendPass("fuse_elewise_add_act_pass");
      // Add a graph viz pass to record a graph.
      if (!strategy.debug_graphviz_path_.empty()) {
        auto viz_pass = AppendPass("graph_viz_pass");
        const std::string graph_path = string::Sprintf(
            "%s%s", strategy.debug_graphviz_path_.c_str(), "_fused_graph");
        viz_pass->Set<std::string>("graph_viz_path",
                                   new std::string(graph_path));
      }
    }

    // Convert graph to run on multi-devices.
    auto multi_devices_pass = AppendPass("multi_devices_pass");
    multi_devices_pass->SetNotOwned<const BuildStrategy>("strategy",
                                                         &strategy_);

    // Add a graph print pass to record a graph with device info.
    if (!strategy_.debug_graphviz_path_.empty()) {
      auto multi_devices_print_pass = AppendPass("multi_devices_print_pass");
      const std::string graph_path =
          string::Sprintf("%s%s", strategy_.debug_graphviz_path_.c_str(),
                          "_multi_devices_graph");
      multi_devices_print_pass->Set<std::string>(kGraphvizPath,
                                                 new std::string(graph_path));
      multi_devices_print_pass->Set<details::GraphvizSSAGraphPrinter>(
          "graph_printer", new details::GraphvizSSAGraphPrinter);
    }

    // Verify that the graph is correct for multi-device executor.
    AppendPass("multi_devices_check_pass");

    if (strategy_.remove_unnecessary_lock_) {
      AppendPass("modify_op_lock_and_record_event_pass");
    }
  }

 private:
  BuildStrategy strategy_;
};

std::shared_ptr<ir::PassBuilder> BuildStrategy::CreatePassesFromStrategy(
    bool finalize_strategy) const {
  if (is_finalized_) {
    return pass_builder_;
  }
  pass_builder_.reset(new ParallelExecutorPassBuilder(*this));
  if (finalize_strategy) {
    is_finalized_ = true;
  }
  return pass_builder_;
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
  // Create a default one if not finalized by user.
  CreatePassesFromStrategy(false);

  std::unique_ptr<ir::Graph> graph(new ir::Graph(main_program));

  // these will transfer ownership to graph. not release by hand.
  std::vector<OpDesc *> *all_op_descs =
      new std::vector<OpDesc *>(main_program.Block(0).AllOps());
  details::GraphNodePool *graph_pool = new details::GraphNodePool;

  graph->Set(details::kGraphNodePool, graph_pool);  // take ownership
  graph->Set(ir::kAllOpDescs, all_op_descs);        // take ownership

  for (std::shared_ptr<ir::Pass> &pass : pass_builder_->AllPasses()) {
    if (pass->Type() == "multi_devices_pass") {
      pass->Erase("places");
      pass->SetNotOwned<const std::vector<platform::Place>>("places", &places);
      pass->Erase("loss_var_name");
      pass->SetNotOwned<const std::string>("loss_var_name", &loss_var_name);
      pass->Erase("params");
      pass->SetNotOwned<const std::unordered_set<std::string>>("params",
                                                               &param_names);
      pass->Erase("local_scopes");
      pass->SetNotOwned<const std::vector<Scope *>>("local_scopes",
                                                    &local_scopes);
#ifdef PADDLE_WITH_CUDA
      platform::NCCLContextMap *nctx = use_cuda ? nccl_ctxs : nullptr;
      pass->Erase("nccl_ctxs");
      pass->SetNotOwned<platform::NCCLContextMap>("nccl_ctxs", nctx);
#endif
    } else if (pass->Type() == "sequential_execution_pass") {
      // may conflict with merge_multi_batch_pass, because it change
      // AllOps inside pass. No pass->Erase(kAllOpDescs) on purpose.
      pass->SetNotOwned(ir::kAllOpDescs, all_op_descs);
    } else if (pass->Type() == "analysis_var_pass") {
      pass->SetNotOwned(details::kGraphNodePool, graph_pool);
    }
    graph = pass->Apply(std::move(graph));
  }
  return graph;
}

}  // namespace details
}  // namespace framework
}  // namespace paddle

USE_PASS(fuse_elewise_add_act_pass);
USE_PASS(graph_viz_pass);
USE_PASS(multi_batch_merge_pass);
USE_PASS(multi_devices_pass);
USE_PASS(multi_devices_check_pass);
USE_PASS(multi_devices_print_pass);
USE_PASS(analysis_var_pass);
USE_PASS(sequential_execution_pass);
USE_PASS(modify_op_lock_and_record_event_pass);
