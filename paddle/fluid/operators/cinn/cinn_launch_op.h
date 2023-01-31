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

#include <chrono>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "cinn/common/target.h"
#include "gflags/gflags.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/paddle2cinn/cinn_compiler.h"
#include "paddle/fluid/operators/cinn/cinn_launch_context.h"
#include "paddle/fluid/operators/cinn/cinn_op_helper.h"
#include "paddle/fluid/platform/profiler.h"

DECLARE_bool(enable_pe_launch_cinn);
DECLARE_bool(enable_interpretercore_launch_cinn);
namespace paddle {
namespace operators {

using CinnCompiler = framework::paddle2cinn::CinnCompiler;
using CinnCompiledObject = framework::paddle2cinn::CinnCompiledObject;

namespace details {

// Tranform Paddle place to CINN target
const ::cinn::common::Target& PlaceToCinnTarget(const platform::Place& place);

// Print detailed compilation result of graph for debug
void DebugCinnCompiledResult(const CinnCompiledObject& result);

// Launch cinn to execute compiled executable program and wait done
void LaunchCinnExecution(const CinnCompiledObject& compiled_obj,
                         const CinnLaunchContext& context,
                         void* stream);

// Set cinn FLAGS (such as FLAGS_cinn_cudnn_deterministic) with paddle's FLAGS.
void SetCinnRuntimeFlags();

}  // namespace details

template <typename DeviceContext, typename T>
class CinnLaunchOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const auto& scope = ctx.scope();
    const auto& place = ctx.GetPlace();
    void* stream = details::GetStream<DeviceContext>(ctx);
    platform::RecordEvent record_event_1(
        "Step 1. Find graph object and prepare input");
    // Step 1. Find graph object and prepare input
    PADDLE_ENFORCE_EQ(ctx.HasAttr(kCompilationKey),
                      true,
                      platform::errors::NotFound(
                          "No Attribute(%s) found for CinnLaunchOp operator.",
                          kCompilationKey));
    const auto& compilation_key = ctx.template Attr<int64_t>(kCompilationKey);
    VLOG(4) << "CinnLaunchOp attribute(" << kCompilationKey << ") "
            << "value:\n"
            << CinnCompiler::GetInstance()->ReadableKey(compilation_key);

    std::map<std::string, const phi::DenseTensor*> inputs_name2tensor;
    std::vector<std::string> input_x_variable_names;
    std::vector<std::string> input_no_need_buffer_variable_names;
    auto add_name2tensor_fn =
        [&inputs_name2tensor](
            const std::vector<std::string>& variable_names,
            const std::vector<const phi::DenseTensor*>& tensors) {
          std::transform(
              variable_names.begin(),
              variable_names.end(),
              tensors.begin(),
              std::inserter(inputs_name2tensor, inputs_name2tensor.end()),
              [](const std::string& name, const phi::DenseTensor* tensor) {
                return std::make_pair(name, tensor);
              });
        };

    auto input_x_tensors = ctx.MultiInput<phi::DenseTensor>(kX);
    if (!input_x_tensors.empty()) {
      input_x_variable_names = std::move(ctx.InputNames(kX));
      add_name2tensor_fn(input_x_variable_names, input_x_tensors);
    }
    auto input_no_need_buffer_tensors =
        ctx.MultiInput<phi::DenseTensor>(kNoNeedBufferX);
    if (!input_no_need_buffer_tensors.empty()) {
      input_no_need_buffer_variable_names =
          std::move(ctx.InputNames(kNoNeedBufferX));
      add_name2tensor_fn(input_no_need_buffer_variable_names,
                         input_no_need_buffer_tensors);
    }

    platform::RecordEvent record_event_2(
        "Step 2. Get compilation result of the graph");
    // Step 2. Get compilation result of the graph
    auto target = details::PlaceToCinnTarget(place);
    using ClockType = std::chrono::steady_clock;
    std::chrono::time_point<ClockType> start_t, end_t;
    if (VLOG_IS_ON(1)) {
      VLOG(1) << "Starts to compile at thread " << std::this_thread::get_id();
      start_t = ClockType::now();
    }
    const auto& cinn_compiled_object = CinnCompiler::GetInstance()->Compile(
        compilation_key, inputs_name2tensor, target, stream);
    if (VLOG_IS_ON(1)) {
      end_t = ClockType::now();
      auto time_sec = std::chrono::duration_cast<std::chrono::milliseconds>(
          end_t - start_t);
      VLOG(1) << "Ends to compile at thread " << std::this_thread::get_id()
              << " , time cost : " << time_sec.count() << " ms";
    }
    details::DebugCinnCompiledResult(cinn_compiled_object);
    auto* launch_context = cinn_compiled_object.launch_context.get();

    platform::RecordEvent record_event_3("Step 3. Set CINN runtime FLAGS.");
    // Step 3. Set CINN runtime FLAGS, such as FLAGS_cinn_cudnn_deterministic.
    details::SetCinnRuntimeFlags();

    // Step 4. Execute the compiled CINN instructions by a PE or
    //         by the CINN compiled program in sequential order
    if (FLAGS_enable_pe_launch_cinn) {
      if (FLAGS_enable_interpretercore_launch_cinn) {
        platform::RecordEvent record_event_4(
            "Step 4. Execute the runtime program by InterpreterCore.");
        VLOG(4) << "Execute the runtime program by InterpreterCore";
        auto* interpreter_core = launch_context->InitializeInterpreterCore(
            place, const_cast<framework::Scope*>(&scope));
        interpreter_core->Run({}, false);
      } else {
        platform::RecordEvent record_event_4(
            "Step 4. Execute the runtime graph by PE.");
        VLOG(4) << "Execute the runtime graph by PE";
        framework::Scope& exec_scope = scope.NewScope();
        auto* pe = launch_context->InitializePE(place, &exec_scope);
        pe->RunWithoutFetch(launch_context->GetSkipEagerVars());
      }
    } else {
      platform::RecordEvent record_event_4(
          "Step 4. Execute the compiled executable program.");
      VLOG(4) << "Execute the compiled executable program";
      launch_context->UpdateCapturedEnv(scope, place);
      LaunchCinnExecution(cinn_compiled_object, *launch_context, stream);
    }
    VLOG(4) << "CinnLaunchOp launch execution done.";
  }
};

}  // namespace operators
}  // namespace paddle
