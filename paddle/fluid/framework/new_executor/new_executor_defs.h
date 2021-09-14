// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/device_event_base.h"
#include "paddle/fluid/platform/event.h"

namespace paddle {
namespace framework {

namespace interpretercore {
static constexpr char kMemcpyH2D[] = "memcpy_h2d";
static constexpr char kMemcpyD2H[] = "memcpy_d2h";
}  // namespace interpretercore

using OpKernelComputeFunc = std::function<void(const ExecutionContext&)>;
using OpKernelMap =
    std::unordered_map<OpKernelType, OpKernelComputeFunc, OpKernelType::Hash>;

struct OpKernelFunc {
  OpKernelComputeFunc compute_func_;
  OperatorBase* operator_base_;
};

struct VariableMetaInfo {
  int var_ref_count_;
  paddle::framework::VarDesc* vardesc_;
};

struct VariableScope {
  std::vector<Variable*> var_list;
  std::map<std::string, int> name2id;
  std::vector<VariableMetaInfo> vec_meta_info_;
};

struct EventRun {
  explicit EventRun(size_t op_id) : op_id_(op_id) {}
  size_t op_id_;
};
struct NextInstruction {
  std::vector<size_t> direct_run_;
  std::vector<EventRun> event_wait_run_;
  std::vector<EventRun> synchronize_run_;
  std::vector<size_t> all_next_ops_;
};

struct EventInter {
  explicit EventInter(size_t var_id,
                      std::shared_ptr<platform::DeviceEvent> event,
                      bool is_sync)
      : var_id_(var_id), event_(event), is_sync_(is_sync) {}
  size_t var_id_;
  std::shared_ptr<platform::DeviceEvent> event_;
  bool is_sync_;
};

struct InstructionInfo {
  std::vector<size_t> dependecy_count_;
};

class RuntimeInferShapeContext;

struct Instruction {
  OpKernelFunc kernel_func_;
  std::shared_ptr<RuntimeContext> runtime_ctx_;
  std::shared_ptr<RuntimeInferShapeContext> infershape_ctx_;
  std::shared_ptr<ExecutionContext> execution_ctx_;
  std::map<std::string, std::vector<int>> input_index_;
  std::map<std::string, std::vector<int>> output_index_;

  std::vector<size_t> gc_check_var_list;
  NextInstruction next_instruction_;

  std::vector<EventInter> intput_events_;
  std::vector<EventInter> output_events_;

  platform::DeviceContext* dev_ctx_;  // not owned

  std::vector<std::pair<Variable*, Variable*>> vec_inplace_in_to_out_;
};

enum class OpFuncType {
  kQueueAsync,  // GPU Kernel or d2h, h2d, send, recv, broadcast
  kQueueSync,   // CPU kernel, block host
};

struct OpFuncNode {
  // int unsed;
  std::map<std::string, std::vector<int>> input_index;
  std::map<std::string, std::vector<int>> output_index;

  OpKernelComputeFunc kernel_func_;
  platform::DeviceContext* dev_ctx_;  // not owned
  OpFuncType type_;
};

}  // namespace framework
}  // namespace paddle
