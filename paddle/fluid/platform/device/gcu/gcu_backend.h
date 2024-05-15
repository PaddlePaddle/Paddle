/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#ifdef PADDLE_WITH_GCU

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/device/gcu/gcu_compiler.h"
#include "paddle/fluid/platform/device/gcu/gcu_info.h"
#include "paddle/fluid/platform/device/gcu/gcu_strategy.h"
#include "paddle/fluid/platform/device/gcu/utils/types.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/timer.h"

namespace paddle {
namespace platform {
namespace gcu {

class GcuBackend {
 public:
  static GcuBackend *GetInstance() {
    GcuBackend *gcu_backend = new GcuBackend();
    return gcu_backend;
  }

  void Compile(const std::vector<const Graph *> &graph_list,
               const std::vector<std::vector<std::string>> &all_feeds,
               const std::vector<std::vector<std::string>> &all_fetches,
               const std::string &program_key);

  void RunGcuOp(const std::vector<const Tensor *> &inputs,
                const std::vector<Tensor *> &outputs,
                const paddle::framework::ExecutionContext &ctx);

  void PostProcess(const std::vector<const Graph *> &before_graph_list,
                   const Graph *post_graph);

  // Sync weights from GCU while training
  void WeightsToHost();

  void SetScope(const Scope &scope) { scope_ = &scope; }
  const Scope *GetScope() { return scope_; }
  void SetGcuStrategy(const GcuStrategy &strategy) { gcu_strategy_ = strategy; }
  const GcuStrategy GetGcuStrategy() { return gcu_strategy_; }

  void InitBackend(bool distributed = false,
                   uint32_t rank_id = 0,
                   int device_id = 0,
                   uint32_t world_size = 1,
                   int32_t node_id = 0);

 public:
  GcuBackend();
  ~GcuBackend();

 private:
  const Scope *scope_ = nullptr;
  GcuStrategy gcu_strategy_;

  std::shared_ptr<GcuCompiler> gcu_compiler_;

  bool is_compiled_ = false;
  std::string target_name_ = "pavo";
};
}  // namespace gcu
}  // namespace platform
}  // namespace paddle
#endif
