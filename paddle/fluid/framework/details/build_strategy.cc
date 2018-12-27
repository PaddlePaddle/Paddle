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

#include "paddle/fluid/framework/details/memory_reuse_types.h"
#include "paddle/fluid/framework/details/multi_devices_graph_check_pass.h"
#include "paddle/fluid/framework/details/multi_devices_graph_print_pass.h"
#include "paddle/fluid/framework/details/reduce_op_handle.h"
#include "paddle/fluid/framework/details/sequential_execution_pass.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/graph_viz_pass.h"

namespace paddle {
namespace framework {
namespace details {

static inline bool SeqOnlyAllReduceOps(const BuildStrategy &strategy) {
  return (!strategy.enable_sequential_execution_ && strategy.num_trainers_ > 1);
}

class ParallelExecutorPassBuilder : public ir::PassBuilder {
 public:
  explicit ParallelExecutorPassBuilder(const BuildStrategy &strategy)
      : ir::PassBuilder(), strategy_(strategy) {
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

    CollectiveContext *context = CollectiveContext::GetInstance();
    context->endpoints_ = strategy_.trainers_endpoints_;
    context->trainer_id_ = strategy_.trainer_id_;
    PADDLE_ENFORCE(strategy_.trainer_id_ >= 0, "trainer_id_ >= 0");
    if (strategy_.trainer_id_ > 0) {
      PADDLE_ENFORCE((unsigned)(strategy_.trainer_id_) <
                         strategy_.trainers_endpoints_.size(),
                     "trainer_id_ < endpoints_ size");
    }
    VLOG(1) << "CollectiveContext:" << context->String();

    // NOTE(dzh): memory optimize should be a runtime pass.
    // However, after multi_devices_pass, VarHandle, OpHandle is
    // the de-fact IR, any reuse on Graph is meaningless.
    // A side-effect of that, memory optimize cannot forsee the fetched vars
    // , so fetchlist should be set persistable before call the Run interface.
    if (strategy.memory_optimize_) {
      auto analysis_var_pass = AppendPass("analysis_var_pass");
    }

    // Convert graph to run on multi-devices.
    if (strategy.reduce_ == BuildStrategy::ReduceStrategy::kAllReduce) {
      auto multi_devices_pass = AppendPass("allreduce_mode_multi_devices_pass");
      multi_devices_pass->SetNotOwned<const BuildStrategy>("strategy",
                                                           &strategy_);
      multi_devices_pass->Set<int>("num_trainers",
                                   new int(strategy_.num_trainers_));
    } else {
      auto multi_devices_pass = AppendPass("reduce_mode_multi_devices_pass");
      multi_devices_pass->SetNotOwned<const BuildStrategy>("strategy",
                                                           &strategy_);
      multi_devices_pass->Set<int>("num_trainers",
                                   new int(strategy_.num_trainers_));
    }

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

    if (SeqOnlyAllReduceOps(strategy)) {
      AppendPass("all_reduce_deps_pass");
    }

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

bool BuildStrategy::IsMultiDevPass(const std::string &pass_name) const {
  // FIXME(zcd) temporary code
  return pass_name == "reduce_mode_multi_devices_pass" ||
         pass_name == "allreduce_mode_multi_devices_pass";
}

std::unique_ptr<ir::Graph> BuildStrategy::Apply(
    const ProgramDesc &main_program, const std::vector<platform::Place> &places,
    const std::string &loss_var_name, const std::vector<Scope *> &local_scopes,
#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
    const bool use_cuda, platform::NCCLContextMap *nccl_ctxs) const {
#else
    const bool use_cuda) const {
#endif
  // Create a default one if not finalized by user.
  CreatePassesFromStrategy(false);

  std::unique_ptr<ir::Graph> graph(new ir::Graph(main_program));
  for (std::shared_ptr<ir::Pass> &pass : pass_builder_->AllPasses()) {
    if (IsMultiDevPass(pass->Type())) {
      pass->Erase("places");
      pass->SetNotOwned<const std::vector<platform::Place>>("places", &places);
      pass->Erase("loss_var_name");
      pass->SetNotOwned<const std::string>("loss_var_name", &loss_var_name);
      pass->Erase("local_scopes");
      pass->SetNotOwned<const std::vector<Scope *>>("local_scopes",
                                                    &local_scopes);
#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
      platform::NCCLContextMap *nctx = use_cuda ? nccl_ctxs : nullptr;
      pass->Erase("nccl_ctxs");
      pass->SetNotOwned<platform::NCCLContextMap>("nccl_ctxs", nctx);
#endif

    } else if (pass->Type() == "analysis_var_pass") {
      const std::vector<OpDesc *> *all_op_descs =
          new std::vector<OpDesc *>(main_program.Block(0).AllOps());
      graph->Set<const std::vector<OpDesc *>>(kAllOpDescs,
                                              all_op_descs);  // take ownership
      graph->Set<GraphNodePool>(kGraphNodePool,
                                new GraphNodePool);  // take ownership

      pass->Erase(kAllOpDescs);
      pass->SetNotOwned<const std::vector<OpDesc *>>(kAllOpDescs, all_op_descs);

    } else if (pass->Type() == "sequential_execution_pass") {
      LOG(INFO) << "set enable_sequential_execution:"
                << enable_sequential_execution_;

      pass->Erase(kAllOpDescs);
      pass->Set<const std::vector<OpDesc *>>(
          kAllOpDescs,
          new std::vector<OpDesc *>(main_program.Block(0).AllOps()));
    } else if (pass->Type() == "all_reduce_deps_pass") {
      LOG(INFO) << "SeqOnlyAllReduceOps:" << SeqOnlyAllReduceOps(*this)
                << ", num_trainers:" << num_trainers_;

      pass->Erase(kAllOpDescs);
      pass->Set<const std::vector<OpDesc *>>(
          kAllOpDescs,
          new std::vector<OpDesc *>(main_program.Block(0).AllOps()));
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
USE_PASS(reduce_mode_multi_devices_pass);
USE_PASS(allreduce_mode_multi_devices_pass);
USE_PASS(multi_devices_check_pass);
USE_PASS(multi_devices_print_pass);
USE_PASS(analysis_var_pass);
USE_PASS(sequential_execution_pass);
USE_PASS(all_reduce_deps_pass);
USE_PASS(modify_op_lock_and_record_event_pass);
