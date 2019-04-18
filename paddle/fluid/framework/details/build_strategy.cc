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
#include <utility>
#include "paddle/fluid/framework/details/memory_optimize_helper.h"
#include "paddle/fluid/framework/details/multi_devices_graph_pass.h"
#include "paddle/fluid/framework/details/multi_devices_graph_print_pass.h"
#include "paddle/fluid/framework/details/reduce_op_handle.h"
#include "paddle/fluid/framework/details/sequential_execution_pass.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/graph_to_program_pass.h"
#include "paddle/fluid/framework/ir/graph_viz_pass.h"

namespace paddle {
namespace framework {
namespace details {

static inline bool SeqOnlyAllReduceOps(const BuildStrategy &strategy) {
  // Should fix the allreduce op order if scheduling
  // them in multiple threads or processes to avoid hang.
  // NOTE: ParallelGraph would execute this pass on each graph, so
  // don't need to append it here.
  return (!strategy.enable_sequential_execution_ &&
          strategy.num_trainers_ > 1) &&
         !strategy.enable_parallel_graph_;
}

class ParallelExecutorPassBuilder : public ir::PassBuilder {
 public:
  explicit ParallelExecutorPassBuilder(const BuildStrategy &strategy)
      : ir::PassBuilder(), strategy_(strategy) {
    // Add a graph viz pass to record a graph.
    if (!strategy_.debug_graphviz_path_.empty()) {
      auto viz_pass = AppendPass("graph_viz_pass");
      const std::string graph_path = string::Sprintf(
          "%s%s", strategy_.debug_graphviz_path_.c_str(), "_original_graph");
      viz_pass->Set<std::string>("graph_viz_path", new std::string(graph_path));
    }

    // Note(zcd): record_skip_memory_opt_vars_pass should be the first pass.
    AppendPass("record_skip_memory_opt_vars_pass");

    if (strategy_.enable_sequential_execution_) {
      VLOG(10) << "Add sequential_execution_pass";
      AppendPass("sequential_execution_pass");
    }

    // Add op fusion.
    if (strategy.sync_batch_norm_) {
      AppendPass("sync_batch_norm_pass");
    }

    // Add op fusion.
    if (strategy.fuse_relu_depthwise_conv_) {
      VLOG(10) << "Add fuse_relu_depthwise_conv_pass";
      AppendPass("fuse_relu_depthwise_conv_pass");
    }

    // NOTE(dzhwinter): A note for automatical inplace.
    // 1. modify program desc passes should put
    // before inplace pass.
    // 2. manually configured inplace should put
    // before inplace_pass

    // Add automatically inplace.
    if (strategy_.enable_inplace_) {
      VLOG(10) << "Add inplace_pass";
      AppendPass("inplace_pass");
    }

    if (strategy_.fuse_elewise_add_act_ops_) {
      VLOG(10) << "Add fuse_elewise_add_act_pass";
      AppendPass("fuse_elewise_add_act_pass");
    }

    // for single card training, fuse_all_reduce_ops is unnecessary.
    // alloc_continuous_space_for_grad_pass should be before of MultiDevPass.
    if (strategy_.fuse_all_reduce_ops_) {
      VLOG(10) << "Add alloc_continuous_space_for_grad_pass";
      AppendPass("alloc_continuous_space_for_grad_pass");
    }

    if (strategy_.fuse_all_optimizer_ops_) {
      if (strategy_.reduce_ == BuildStrategy::ReduceStrategy::kReduce ||
          strategy_.is_distribution_) {
        VLOG(3)
            << "Currently, fuse_all_optimizer_ops only works under AllReduce "
               "mode.";
        strategy_.fuse_all_optimizer_ops_ = false;
      } else {
        VLOG(10) << "Add alloc_continuous_space_for_grad_pass";
        AppendPass("alloc_continuous_space_for_grad_pass");
        // NOTE: fuse_all_xx_ops will count the number of xx operator first,
        // if the number is zero, fuse_all_reduce_ops will do nothing.
        // Currently, only one type of optimization algorithm can be fused.
        VLOG(10) << "Add fuse_adam_op_pass";
        AppendPass("fuse_adam_op_pass");
        VLOG(10) << "Add fuse_sgd_op_pass";
        AppendPass("fuse_sgd_op_pass");
      }
    }

    // Add a graph viz pass to record a graph.
    if (!strategy.debug_graphviz_path_.empty()) {
      auto viz_pass = AppendPass("graph_viz_pass");
      const std::string graph_path = string::Sprintf(
          "%s%s", strategy_.debug_graphviz_path_.c_str(), "_fused_graph");
      viz_pass->Set<std::string>("graph_viz_path", new std::string(graph_path));
    }

    CollectiveContext *context = CollectiveContext::GetInstance();
    context->endpoints_ = strategy_.trainers_endpoints_;
    context->trainer_id_ = strategy_.trainer_id_;
    PADDLE_ENFORCE(strategy_.trainer_id_ >= 0, "trainer_id_ >= 0");
    if (strategy_.trainer_id_ > 0 && strategy_.trainers_endpoints_.size() > 0) {
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
    if (strategy_.memory_optimize_) {
      VLOG(10) << "Add memory_optimize_pass";
      AppendPass("memory_optimize_pass");
    }

    AppendMultiDevPass(strategy_);

    if (strategy_.fuse_all_reduce_ops_) {
      // NOTE: fuse_all_reduce_ops will count the number of all_reduce operator
      // first, if the number is zero, fuse_all_reduce_ops will do nothing.
      VLOG(10) << "Add fuse_all_reduce_op_pass";
      AppendPass("fuse_all_reduce_op_pass");
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

    // experimental shows that the program will be faster if append
    // all_reduce_deps_pass here.
    if (!strategy_.enable_parallel_graph_ &&
        (SeqOnlyAllReduceOps(strategy_) ||
         strategy.reduce_ == BuildStrategy::ReduceStrategy::kAllReduce)) {
      AppendPass("all_reduce_deps_pass");
    }

    if (strategy_.remove_unnecessary_lock_) {
      VLOG(10) << "Add modify_op_lock_and_record_event_pass";
      AppendPass("modify_op_lock_and_record_event_pass");
    }

    // Verify that the graph is correct for multi-device executor.
    AppendPass("multi_devices_check_pass");
  }

  // Convert graph to run on multi-devices.
  void AppendMultiDevPass(const BuildStrategy &strategy) {
    ir::Pass *multi_devices_pass = nullptr;
    if (strategy.is_distribution_) {
      VLOG(10) << "Add dist_multi_devices_pass";
      multi_devices_pass = AppendPass("dist_multi_devices_pass").get();
    } else {
      if (strategy.reduce_ == BuildStrategy::ReduceStrategy::kAllReduce) {
        VLOG(10) << "Add all_reduce_mode_multi_devices_pass";
        multi_devices_pass =
            AppendPass("all_reduce_mode_multi_devices_pass").get();
      } else if (strategy.reduce_ == BuildStrategy::ReduceStrategy::kReduce) {
        VLOG(10) << "Add reduce_mode_multi_devices_pass";
        multi_devices_pass = AppendPass("reduce_mode_multi_devices_pass").get();
      } else {
        PADDLE_THROW("Unknown reduce strategy.");
      }
    }
    multi_devices_pass->SetNotOwned<const BuildStrategy>("strategy",
                                                         &strategy_);
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
  return framework::details::MultiDevSSAGraphBuilder().count(pass_name) > 0;
}

ir::Graph *BuildStrategy::Apply(ir::Graph *graph,
                                const std::vector<platform::Place> &places,
                                const std::string &loss_var_name,
                                const std::vector<Scope *> &local_scopes,
                                const size_t &nranks,
#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
                                const bool use_cuda,
                                platform::NCCLContextMap *nccl_ctxs) const {
#else
                                const bool use_cuda) const {
#endif
  // Create a default one if not finalized by user.
  CreatePassesFromStrategy(false);

  for (std::shared_ptr<ir::Pass> &pass : pass_builder_->AllPasses()) {
    if (IsMultiDevPass(pass->Type())) {
      pass->Erase(kPlaces);
      pass->SetNotOwned<const std::vector<platform::Place>>(kPlaces, &places);
      pass->Erase(kLossVarName);
      pass->SetNotOwned<const std::string>(kLossVarName, &loss_var_name);
      pass->Erase(kLocalScopes);
      pass->SetNotOwned<const std::vector<Scope *>>(kLocalScopes,
                                                    &local_scopes);
      pass->Erase(kNRanks);
      pass->Set<size_t>(kNRanks, new size_t(nranks));

#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
      platform::NCCLContextMap *nctx = use_cuda ? nccl_ctxs : nullptr;
      pass->Erase(kNCCLCtxs);
      pass->SetNotOwned<platform::NCCLContextMap>(kNCCLCtxs, nctx);
#endif
    } else if (pass->Type() == "alloc_continuous_space_for_grad_pass" ||
               pass->Type() == "fuse_adam_op_pass" ||
               pass->Type() == "fuse_sgd_op_pass" ||
               pass->Type() == "fuse_all_reduce_op_pass") {
      pass->Erase(kPlaces);
      pass->SetNotOwned<const std::vector<platform::Place>>(kPlaces, &places);
      pass->Erase(kLocalScopes);
      pass->SetNotOwned<const std::vector<Scope *>>(kLocalScopes,
                                                    &local_scopes);
      if (pass->Type() == "fuse_all_reduce_op_pass") {
#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
        platform::NCCLContextMap *nctx = use_cuda ? nccl_ctxs : nullptr;
        pass->Erase(kNCCLCtxs);
        pass->SetNotOwned<platform::NCCLContextMap>(kNCCLCtxs, nctx);
#endif
      }
    } else if (pass->Type() == "alloc_continuous_space_for_grad_pass") {
      pass->Erase(kPlaces);
      pass->SetNotOwned<const std::vector<platform::Place>>(kPlaces, &places);
      pass->Erase(kLocalScopes);
      pass->SetNotOwned<const std::vector<Scope *>>(kLocalScopes,
                                                    &local_scopes);
    } else if (pass->Type() == "sequential_execution_pass") {
      LOG(INFO) << "set enable_sequential_execution:"
                << enable_sequential_execution_;
    } else if (pass->Type() == "all_reduce_deps_pass") {
      LOG(INFO) << "SeqOnlyAllReduceOps:" << SeqOnlyAllReduceOps(*this)
                << ", num_trainers:" << num_trainers_;
    } else if (pass->Type() == "fuse_relu_depthwise_conv_pass") {
      if (!use_cuda) {
        LOG(WARNING) << "fuse_relu_depthwise_conv_pass is only supported on "
                        "GPU, skipped.";
        continue;
      }
    }
    VLOG(3) << "Start Apply Pass " << pass->Type();
    graph = pass->Apply(graph);
    VLOG(3) << "Finish Apply Pass " << pass->Type();
  }
  return graph;
}

}  // namespace details
}  // namespace framework
}  // namespace paddle

USE_PASS(sync_batch_norm_pass);
USE_PASS(fuse_relu_depthwise_conv_pass);
USE_PASS(fuse_elewise_add_act_pass);
USE_PASS(graph_viz_pass);
USE_PASS(multi_batch_merge_pass);
USE_PASS(reduce_mode_multi_devices_pass);
USE_PASS(all_reduce_mode_multi_devices_pass);
USE_PASS(dist_multi_devices_pass);
USE_PASS(multi_devices_check_pass);
USE_PASS(multi_devices_print_pass);
USE_PASS(memory_optimize_pass);
USE_PASS(sequential_execution_pass);
USE_PASS(all_reduce_deps_pass);
USE_PASS(modify_op_lock_and_record_event_pass);
USE_PASS(inplace_pass);
USE_PASS(lock_free_optimize_pass);
USE_PASS(alloc_continuous_space_for_grad_pass);
USE_PASS(graph_to_program_pass);
USE_PASS(fuse_adam_op_pass);
USE_PASS(fuse_sgd_op_pass);
USE_PASS(fuse_all_reduce_op_pass);
USE_PASS(record_skip_memory_opt_vars_pass);
