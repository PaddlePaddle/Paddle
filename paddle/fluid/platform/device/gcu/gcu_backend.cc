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

#include "paddle/fluid/platform/device/gcu/gcu_backend.h"

#include <memory>
#include <set>
#include <vector>

#include "dtu_sdk/dtu_sdk.h"
#include "paddle/fluid/platform/device/gcu/gcu_strategy.h"
#include "paddle/fluid/platform/device/gcu/runtime/gcu_rt_interface.h"

namespace paddle {
namespace platform {
namespace gcu {
GcuBackend::GcuBackend() {
  ::dtu::DTUSDKInit();
  target_name_ = runtime::Context::GlobalTargetName();
}

GcuBackend::~GcuBackend() { ::dtu::DTUSDKFini(); }

// Sync weights from GCU while training
void GcuBackend::WeightsToHost() {}

void GcuBackend::Compile(
    const std::vector<const Graph *> &graph_list,
    const std::vector<std::vector<std::string>> &all_feeds,
    const std::vector<std::vector<std::string>> &all_fetches,
    const std::string &program_key) {
  bool check = (graph_list.size() == all_feeds.size()) &&
               (graph_list.size() == all_fetches.size()) &&
               (graph_list.size() == 1 || graph_list.size() == 3);
  PADDLE_ENFORCE_EQ(
      check,
      true,
      platform::errors::InvalidArgument(
          "Invalied input for compile, params size:[%zu, %zu, %zu]",
          graph_list.size(),
          all_feeds.size(),
          all_fetches.size()));
  VLOG(2) << "Enter GcuBackend::Compile, graph size:" << graph_list.size()
          << ", program_key:" << program_key;
  gcu_compiler_->Compile(graph_list, all_feeds, all_fetches, program_key);
}

void GcuBackend::PostProcess(
    const std::vector<const Graph *> &before_graph_list,
    const Graph *post_graph) {
  gcu_compiler_->PostProcess(before_graph_list, post_graph);
}

void GcuBackend::InitBackend(bool distributed,
                             uint32_t rank_id,
                             int device_id,
                             uint32_t world_size,
                             int32_t node_id) {
  auto rt_info = std::make_shared<runtime::GcuRunTimeInfo>(
      device_id, distributed, rank_id, world_size, node_id);
  PADDLE_ENFORCE_NE(rt_info,
                    nullptr,
                    platform::errors::PreconditionNotMet(
                        "Failed to create rt_info on GCU rank:%u, device_id:%d",
                        rank_id,
                        device_id));
  runtime::GcuSetCurrentDevice(device_id);
  runtime::GcuSetRuntimeInfo(device_id, rt_info);

  VLOG(3) << "InitBackend rank_id:" << rank_id << ", device_id:" << device_id
          << ", world_size:" << world_size << ", node_id:" << node_id
          << ", distributed:" << distributed;

  gcu_compiler_ = std::make_shared<GcuCompiler>();
}

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
