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
#include <unordered_set>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/ir/pass_builder.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"

#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
#include "paddle/fluid/platform/nccl_helper.h"
#endif

namespace paddle {
namespace framework {
namespace details {

struct BuildStrategy {
  // ParallelExecutor supports two modes of ReduceStrategy, kAllReduce and
  // kReduce, for CPU and GPU. If you use kAllReduce, different threads
  // optimize their parameters separately. If you use kReduce, the optimizations
  // of parameters are distributed to different threads.
  // For example, a model has 100 parameters and is running with four threads,
  // if you choose kAllReduce, every thread is to optimize 100 parameters
  // separately, if you choose kReduce, every thread is to optimize 25
  // parameters.
  // Of particular note is, if you use kReduce when using CPU training,
  // all the parameters are shared between different threads. This feature will
  // save memory.
  // FIXME(zcd): The result of the two modes(kAllReduce and kReduce) maybe not
  // equal for GPU. Because, the result of the different order of summing maybe
  // different, for example, the result of `a+b+c+d` may be different with the
  // result of `c+a+b+d`.
  // For GPU, the implementation of kAllReduce and kReduce is adopted NCCL,
  // so the result of kAllReduce and kReduce maybe not equal.
  // For CPU, if you want to fix the order of summing to make the result
  // of kAllReduce and kReduce no diff, you can add
  // `FLAGS_cpu_deterministic=true` to env.
  enum class ReduceStrategy { kAllReduce = 0, kReduce = 1 };

  enum class GradientScaleStrategy {
    kCoeffNumDevice = 0,
    kOne = 1,
    // user can customize gradient scale to use, and just feed
    // it into exe.run().
    kCustomized = 2,
  };

  ReduceStrategy reduce_{ReduceStrategy::kAllReduce};
  GradientScaleStrategy gradient_scale_{GradientScaleStrategy::kCoeffNumDevice};

  std::string debug_graphviz_path_{""};

  // Add dependency between backward ops and optimization ops, make sure that
  // all the backward ops are finished before running the optimization ops.
  // It might make the training speed of data parallelism faster.
  bool enable_backward_optimizer_op_deps_{true};
  // TODO(dev-paddle): enable_sequential_execution depends on
  // kStaleProgramOpDescs, it is not appropriate, because kStaleProgramOpDescs
  // will be removed in the near future.
  bool enable_sequential_execution_{false};
  bool remove_unnecessary_lock_{true};
  // TODO(dev-paddle): cache_runtime_context may cause some models to hang up
  // while running.
  bool cache_runtime_context_{false};

  // Operator fusion
  // TODO(dev-paddle): fuse_elewise_add_act_ops may cause some models have
  // cycle.
  bool fuse_elewise_add_act_ops_{false};
  // Fuse_all_optimizer_ops and fuse_all_reduce_ops require that gradients
  // should not be sparse types
  bool fuse_all_optimizer_ops_{false};
  bool fuse_all_reduce_ops_{false};
  // fuse_relu_depthwise_conv can fuse the `relu ->
  // depthwise_conv`
  bool fuse_relu_depthwise_conv_{false};
  // NOTE(zcd): In reduce mode, fusing broadcast ops may make the program
  // faster. Because fusing broadcast OP equals delaying the execution of all
  // broadcast Ops, in this case, all nccl streams are used only for reduce
  // operations for a period of time.
  bool fuse_broadcast_ops_{false};
  // replace batch_norm with sync_batch_norm.
  bool sync_batch_norm_{false};

  // mkldnn_enabled_op_types specify the operator type list to
  // use MKLDNN acceleration. It is null in default, means
  // that all the operators supported by MKLDNN will be
  // accelerated. And it should not be set when
  // FLAGS_use_mkldnn=false
  std::unordered_set<std::string> mkldnn_enabled_op_types_;

  // FIXME(liuwei1031) disable memory_optimzie and enable_inplace in 1.4
  // to open them by default, we need to solve the fetch variable issue
  // TODO(liuwei1031): memory_optimize depends on kStaleProgramOpDescs,
  // it is not appropriate, because kStaleProgramOpDescs will be removed in the
  // near future.
  bool memory_optimize_{false};

  // Turn on inplace by default.
  bool enable_inplace_{true};

  // TODO(zjl): Remove this flag when MemoryOptimizePass is refactored
  bool use_legacy_memory_optimize_strategy_{false};

  // FIXME(zcd): is_distribution_ is a temporary field, because in pserver mode,
  // num_trainers is 1, so the current fields of build_strategy doesn't tell if
  // it's distributed model.
  bool is_distribution_{false};
  bool async_mode_{false};
  int num_trainers_{1};
  int trainer_id_{0};
  std::vector<std::string> trainers_endpoints_;

  // NCCL config
  size_t nccl_comm_num_{1};
  // The picture is here:
  // https://github.com/PaddlePaddle/Paddle/pull/17263#discussion_r285411396
  bool use_hierarchical_allreduce_{false};
  // Nccl ranks in a node when use hierarchical allreduce, it's setted to gpu
  // cards' number in most cases.
  size_t hierarchical_allreduce_inter_nranks_{0};
  // Nccl ranks bewteen nodes when use hierarchical allreduce, it's setted to
  // nodes number.
  size_t hierarchical_allreduce_exter_nranks_{0};

  // NOTE:
  // Before you add new options, think if it's a general strategy that works
  // with other strategy. If not, the strategy should be created through
  // CreatePassesFromStrategy and the pass can be managed separately.

  // User normally doesn't need to call this API.
  // The PassBuilder allows for more customized insert, remove of passes
  // from python side.
  // A new PassBuilder is created based on configs defined above and
  // passes are owned by the PassBuilder.
  std::shared_ptr<ir::PassBuilder> CreatePassesFromStrategy(
      bool finalize_strategy) const;

  bool IsFinalized() const { return is_finalized_; }

  bool IsMultiDevPass(const std::string &pass_name) const;

  // Apply the passes built by the pass_builder_. The passes will be
  // applied to the Program and output an ir::Graph.
  ir::Graph *Apply(ir::Graph *graph, const std::vector<platform::Place> &places,
                   const std::string &loss_var_name,
                   const std::vector<Scope *> &local_scopes,
                   const size_t &nranks,
#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
                   const bool use_cuda,
                   platform::NCCLCommunicator *nccl_ctxs) const;
#else
                   const bool use_cuda) const;
#endif

  // If set true, ParallelExecutor would build the main_program into multiple
  // graphs,
  // each of the graphs would run with one device. This approach can achieve
  // better performance
  // on some scenarios.
  mutable bool enable_parallel_graph_ = false;

 private:
  mutable bool is_finalized_ = false;
  mutable std::shared_ptr<ir::PassBuilder> pass_builder_;
};

}  // namespace details
}  // namespace framework
}  // namespace paddle
