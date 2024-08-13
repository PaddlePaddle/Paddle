/* Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

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

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/details/build_strategy.h"
#include "paddle/fluid/framework/feed_fetch_type.h"
#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/phi/core/platform/device_context.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/fluid/platform/device/gpu/nccl_helper.h"
#endif

namespace paddle {
namespace framework {

class CompiledProgramPrivate;

using details::BuildStrategy;
namespace p = paddle::platform;
using DeviceType = paddle::platform::DeviceType;

class CompiledProgram {
  DISABLE_COPY_AND_ASSIGN(CompiledProgram);

 public:
  TEST_API explicit CompiledProgram(const std::vector<phi::Place> &places,
                                    const std::vector<std::string> &bcast_vars,
                                    const std::string &loss_var_name,
                                    Scope *scope,
                                    const std::vector<Scope *> &local_scopes,
                                    const BuildStrategy &build_strategy,
                                    ir::Graph *graph);

  TEST_API ~CompiledProgram();

  std::vector<Scope *> &GetLocalScopes();

 private:
  // broadcast the parameters from the 0th device.
  // trainer_id the trainer index in nccl distributed training.
  void BCastParamsToDevices(const std::vector<std::string> &vars,
                            int trainer_id = 0) const;

  void InitProgramPrivateMemberInfo(const BuildStrategy &build_strategy,
                                    size_t device_count);

  void InitReaderQueueDeviceCount(ir::Graph *graph,
                                  const Scope &scope,
                                  size_t dev_cnt);

  void CreateLocalScopes(Scope *global_scope,
                         const std::vector<Scope *> &local_scopes,
                         bool create_new);

  std::vector<ir::Graph *> CloneGraphToMultiDevices(ir::Graph *graph);

  void PrepareNCCLCommunicator(Scope *global_scope);

  std::vector<ir::Graph *> CompileGraphWithBuildStrategy(
      ir::Graph *graph,
      std::vector<ir::Graph *> *graphs,
      const std::string &loss_var_name);

  CompiledProgramPrivate *member_;
};
}  // namespace framework
}  // namespace paddle
