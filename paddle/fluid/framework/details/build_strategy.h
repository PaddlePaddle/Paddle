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
    kCustomized = 2,
  };

  enum class OptimizeStrategy {
    // To be Implemented,bruteforce, recursive compute unused var names.
    kBruteForce = 0,
    kControlFlowGraph = 1,  // use cfg_graph algorithm, faster speed.
  };

  ReduceStrategy reduce_{ReduceStrategy::kAllReduce};
  GradientScaleStrategy gradient_scale_{GradientScaleStrategy::kCoeffNumDevice};
  OptimizeStrategy strategy_{OptimizeStrategy::kControlFlowGraph};

  std::string debug_graphviz_path_{""};

  bool fuse_elewise_add_act_ops_{false};

  bool fuse_all_optimizer_ops_{false};

  bool fuse_all_reduce_ops_{false};

  bool fuse_relu_depthwise_conv_{false};

  bool sync_batch_norm_{false};

  // FIXME(liuwei1031) disable memory_optimzie and enable_inplace in 1.4
  // to open them by default, we need to solve the fetch variable issue
  bool memory_optimize_{false};

  bool enable_inplace_{false};

  // NOTE(zjl): replace relu with relu2 with outputing an extra mask
  // to save memory.
  bool relu_memory_optimize_{true};

  bool enable_sequential_execution_{false};

  // NOTE(zcd): In reduce mode, fusing broadcast ops may make the program
  // faster. Because fusing broadcast OP equals delaying the execution of all
  // broadcast Ops, in this case, all nccl streams are used only for reduce
  // operations for a period of time.
  bool fuse_broadcast_ops_{false};

  // FIXME(zcd): is_distribution_ is a temporary field, because in pserver mode,
  // num_trainers is 1, so the current fields of build_strategy doesn't tell if
  // it's distributed model.
  bool is_distribution_{false};
  bool async_mode_{false};
  int num_trainers_{1};
  int trainer_id_{0};
  std::vector<std::string> trainers_endpoints_;
  bool remove_unnecessary_lock_{true};

  bool cache_runtime_context_{false};
  bool cache_expected_kernel_{true};

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
                   platform::NCCLContextMap *nccl_ctxs) const;
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
