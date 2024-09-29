// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "paddle/common/flags.h"

#include "paddle/fluid/framework/details/exception_holder.h"
#include "paddle/fluid/framework/new_executor/garbage_collector/garbage_collector.h"
#include "paddle/fluid/framework/new_executor/interpreter/dependency_builder.h"
#include "paddle/fluid/framework/new_executor/interpreter/execution_config.h"
#include "paddle/fluid/framework/new_executor/interpreter/interpreter_util.h"
#include "paddle/fluid/framework/new_executor/interpreter/stream_analyzer.h"
#include "paddle/fluid/framework/new_executor/new_executor_defs.h"
#include "paddle/fluid/framework/new_executor/profiler.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/phi/backends/device_manager.h"
#include "paddle/phi/core/memory/allocation/spin_lock.h"
#include "paddle/phi/core/platform/device_event.h"

COMMON_DECLARE_bool(new_executor_serial_run);
PD_DECLARE_bool(new_executor_static_build);
PD_DECLARE_bool(new_executor_use_inplace);
PD_DECLARE_bool(new_executor_use_local_scope);

COMMON_DECLARE_bool(check_nan_inf);
COMMON_DECLARE_bool(benchmark);
COMMON_DECLARE_uint64(executor_log_deps_every_microseconds);
COMMON_DECLARE_bool(new_executor_use_cuda_graph);
COMMON_DECLARE_bool(enable_pir_in_executor);
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
COMMON_DECLARE_bool(sync_nccl_allreduce);
#endif

constexpr const char* kExceptionCaught = "ExceptionCaught";
constexpr const char* kTaskCompletion = "TaskCompletion";

namespace paddle {
namespace framework {

/// @brief InterpreterBaseImpl is a abstract Base Class and define necessary
/// interface with virtual keywords for Derived class.
/// TODO(Aurelius84): Clean unnecessary interface to keep cohesion.
class InterpreterBaseImpl {
 public:
  virtual ~InterpreterBaseImpl() = default;
  virtual paddle::framework::FetchList Run(
      const std::vector<std::string>& feed_names,
      const std::vector<phi::DenseTensor>& feed_tensors,
      bool need_fetch = true,
      bool enable_job_schedule_profiler = false,
      bool switch_stream = false) = 0;

  virtual paddle::framework::FetchList Run(
      const std::vector<std::string>& feed_names,
      bool need_fetch = true,
      bool enable_job_schedule_profiler = false,
      bool enable_op_profiling = false,
      bool switch_stream = false) = 0;

  virtual void ShareWorkQueueFrom(InterpreterBaseImpl* src) = 0;

  virtual void ShareBuildResultsFrom(const InterpreterBaseImpl& src) = 0;

  virtual void SetCopyProgram(std::shared_ptr<ProgramDesc> prog) = 0;

  virtual std::shared_ptr<ProgramDesc> GetMutableCopyProgram() = 0;

  virtual void SetSkipGcVars(const std::set<std::string>& skip_gc_vars) = 0;

  virtual const std::set<std::string>& JitInputVars() const = 0;

  virtual void SetJitInputVars(const std::set<std::string>& jit_input_vars) = 0;

  virtual const VariableScope* GetVariableScope() const = 0;

  virtual void reset_scope(Scope* new_scope) = 0;

  virtual const Scope* local_scope() const = 0;

  virtual const phi::Place& GetPlace() const = 0;

  virtual void SetOutputHooks(const std::vector<HookFunc>& hookfuncs) = 0;

  virtual void SetInputHooks(const std::vector<HookFunc>& hookfuncs) = 0;

  virtual void SetOutputHooks(const std::vector<PirHookFunc>& hookfuncs) = 0;

  virtual void SetInputHooks(const std::vector<PirHookFunc>& hookfuncs) = 0;

  virtual std::shared_ptr<std::vector<size_t>> GetDependencyCount() const = 0;

  virtual bool IsSharedResultsBuild() const = 0;

  virtual void Build(const std::vector<std::string>& feed_names,
                     std::vector<paddle::framework::OpFuncNode>* op_func_nodes,
                     bool switch_stream = false) = 0;

  virtual bool IsStaticBuild() const = 0;

  virtual std::tuple<double, double> InterpreterRunTime() = 0;

  // Only for debug
  virtual Variable* DebugVar(const std::string& name) const = 0;
};

inline void SetDeviceId(const phi::Place& place) {
  // TODO(zhiqiu): reduce the cost
  if (phi::is_gpu_place(place)) {
#if !defined(PADDLE_WITH_CUDA) && !defined(PADDLE_WITH_HIP)
    PADDLE_THROW(common::errors::Unavailable(
        "Cannot run operator on place %s, please recompile paddle or "
        "reinstall Paddle with CUDA support.",
        place));
#else
    auto dev_id = place.device;
    platform::SetDeviceId(dev_id);
#endif
  } else if (phi::is_xpu_place(place)) {
#ifndef PADDLE_WITH_XPU
    PADDLE_THROW(common::errors::Unavailable(
        "Cannot run operator on place %s, please recompile paddle or "
        "reinstall Paddle with XPU support.",
        place));
#else
    auto dev_id = place.device;
    platform::SetXPUDeviceId(dev_id);
#endif
  } else if (phi::is_custom_place(place)) {
#ifndef PADDLE_WITH_CUSTOM_DEVICE
    PADDLE_THROW(common::errors::Unavailable(
        "Cannot run operator on place %s, please recompile paddle or "
        "reinstall Paddle with CustomDevice support.",
        place));
#else
    phi::DeviceManager::SetDevice(place);
#endif
  }
}

}  // namespace framework
}  // namespace paddle
