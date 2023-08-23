// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/auto_schedule/measure/simple_runner.h"

#include <algorithm>
#include <chrono>
#include <iterator>
#include <limits>
#include <memory>
#include <random>

#include "paddle/cinn/common/target.h"
#include "paddle/cinn/hlir/framework/buffer.h"
#include "paddle/cinn/hlir/framework/scope.h"
#include "paddle/cinn/hlir/framework/tensor.h"

namespace cinn {
namespace auto_schedule {

using hlir::framework::Buffer;
using hlir::framework::Shape;
using hlir::framework::Tensor;

// Parameters that needs to be initialized to 0.
// Key is the Op name, and value is the index of the input parameter in the Op.
static const std::unordered_map<std::string, std::vector<int>>
    kInitWithZeroParams = {
        {"lookup_table", {1}},
        {"gather", {1}},
        {"gather_nd", {1}},
        {"scatter_assign", {2}},
        {"scatter_add", {2}},
};

// Generate random value and populate them to the output address of memory
static void PopulateRandomValue(const common::Type& type,
                                const int numel,
                                void* raw_ptr) {
  std::random_device seed;
  std::default_random_engine engine(seed());

  if (type == common::Bool()) {
    auto* fmt_ptr = reinterpret_cast<bool*>(raw_ptr);
    std::bernoulli_distribution dist(0.5);
    std::generate_n(
        fmt_ptr, numel, [&engine, &dist]() { return dist(engine); });
  } else if (type == common::I32()) {
    auto* fmt_ptr = reinterpret_cast<int*>(raw_ptr);
    std::uniform_int_distribution<int> dist(std::numeric_limits<int>::min(),
                                            std::numeric_limits<int>::max());
    std::generate_n(
        fmt_ptr, numel, [&engine, &dist]() { return dist(engine); });
  } else if (type == common::I64()) {
    auto* fmt_ptr = reinterpret_cast<int64_t*>(raw_ptr);
    std::uniform_int_distribution<int64_t> dist(
        std::numeric_limits<int64_t>::min(),
        std::numeric_limits<int64_t>::max());
    std::generate_n(
        fmt_ptr, numel, [&engine, &dist]() { return dist(engine); });
  } else if (type == common::F32()) {
    auto* fmt_ptr = reinterpret_cast<float*>(raw_ptr);
    std::uniform_real_distribution<float> dist(
        std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
    std::generate_n(
        fmt_ptr, numel, [&engine, &dist]() { return dist(engine); });
  } else {
    CHECK_EQ(type.bytes(), 8)
        << "Unsupported type: " << type << ", type.bytes = " << type.bytes();
    auto* fmt_ptr = reinterpret_cast<uint8_t*>(raw_ptr);
    std::uniform_int_distribution<uint8_t> dist(
        std::numeric_limits<uint8_t>::min(),
        std::numeric_limits<uint8_t>::max());
    std::generate_n(
        fmt_ptr, numel, [&engine, &dist]() { return dist(engine); });
  }
}

// Initialize a tensor with 0 if init_with_zero == true, otherwise initialize
// the tensor with random value.
static void InitTensorData(Tensor tensor,
                           const common::Target& target,
                           bool init_with_zero) {
  int mem_size = tensor->shape().numel() * tensor->type().bytes();
  auto* tensor_data = tensor->mutable_data(target, tensor->type());
#ifdef CINN_WITH_CUDA
  if (target == common::DefaultNVGPUTarget()) {
    if (init_with_zero) {
      cudaMemset(tensor_data, 0, mem_size);
    } else {
      void* tmp_buffer = malloc(mem_size);
      PopulateRandomValue(tensor->type(), tensor->shape().numel(), tmp_buffer);
      cudaMemcpy(tensor_data, tmp_buffer, mem_size, cudaMemcpyHostToDevice);
      free(tmp_buffer);
    }
  }
#endif
  if (target == common::DefaultHostTarget()) {
    if (init_with_zero) {
      memset(tensor_data, 0, mem_size);
    } else {
      PopulateRandomValue(tensor->type(), tensor->shape().numel(), tensor_data);
    }
  }
}

// Find all parameter names in the task corresponding to the MeasureInput
// that need to be initialized to 0 when measuring.
static std::unordered_set<std::string> ParamsNeedInitWithZero(
    const MeasureInput& input) {
  std::unordered_set<std::string> res;
  std::vector<hlir::framework::Node*> nodes =
      input.task->subgraph->CollectNodes();
  for (auto* node : nodes) {
    if (kInitWithZeroParams.count(node->op()->name) != 0) {
      std::vector<int> param_idxs = kInitWithZeroParams.at(node->op()->name);
      const auto& inlinks = node->inlinks_in_order();
      for (int param_idx : param_idxs) {
        CHECK_GT(inlinks.size(), param_idx);
        auto& edge = inlinks.at(param_idx);
        std::string param_name =
            edge->source()->as<hlir::framework::NodeData>()->id();
        VLOG(6) << "param needs to be init with 0: " << param_name;
        res.insert(param_name);
      }
    }
  }

  return res;
}

SimpleRunner::SimpleRunner(int repeat_times) : repeat_times_(repeat_times) {
  CHECK_GT(repeat_times_, 0) << "repeat_times can't less than 0";
}

// Prepare execution arguments of all instructions to run, a argument
// may be obtained from the input of measurement or allocating new buffer
// with random value.
std::map<std::string, cinn_pod_value_t> SimpleRunner::PrepareArgs(
    const MeasureInput& input,
    const BuildResult& build_result,
    hlir::framework::Scope* temp_scope) {
  std::map<std::string, cinn_pod_value_t> result;

  const auto& target = input.task->target;
  const auto* input_args = input.execution_args;
  const auto* compiled_scope = build_result.compiled_scope;
  const auto& instructions = build_result.runtime_program->GetRunInstructions();

  std::unordered_set<std::string> params_need_init_with_zero =
      ParamsNeedInitWithZero(input);

  auto fill_arg_fn = [&](const std::string& param) {
    VLOG(6) << "Filling argument:" << param;
    // the argument is duplicated and has been prepared.
    if (result.count(param)) {
      return;
    }

    // if the input of measurement specifies this argument,
    // we should use it firstly.
    if (input_args && input_args->count(param)) {
      VLOG(6) << "Argument[" << param << "] use input value";
      result.emplace(param, input_args->at(param));
      return;
    }

    if (temp_scope->FindVar(param)) {
      auto temp_tensor = temp_scope->GetTensor(param);
      result.emplace(param, temp_tensor->buffer());
      return;
    }

    // allocate a new buffer for this argument and store it in
    // the temporary scope to be released at proper time.
    auto compiled_tensor = compiled_scope->GetTensor(param);
    temp_scope->Var<Tensor>(param);
    auto temp_tensor = temp_scope->GetTensor(param);
    temp_tensor->Resize(compiled_tensor->shape());
    temp_tensor->set_type(compiled_tensor->type());
    temp_tensor->mutable_data(target, compiled_tensor->type());
    InitTensorData(
        temp_tensor, target, params_need_init_with_zero.count(param) != 0);

    result.emplace(param, temp_tensor->buffer());
  };

  for (auto&& instr : instructions) {
    for (auto&& args : instr->GetInArgs()) {
      std::for_each(args.begin(), args.end(), fill_arg_fn);
    }

    for (auto&& args : instr->GetOutArgs()) {
      std::for_each(args.begin(), args.end(), fill_arg_fn);
    }
  }
  return result;
}

MeasureResult SimpleRunner::Run(const MeasureInput& input,
                                const BuildResult& build_result) {
  MeasureResult result;
  auto t_start = std::chrono::steady_clock::now();
  // prepare execution arguments
  VLOG(4) << "SimpleRunner prepare execution arguments";
  hlir::framework::Scope temp_scope;  // used for store temporary allocated data
  auto execution_args = PrepareArgs(input, build_result, &temp_scope);

  // Execute each instruction repeatedly and take the average as cost.
  result.execution_cost = 0;
  const auto& instructions = build_result.runtime_program->GetRunInstructions();
  for (auto ct = 0; ct < instructions.size(); ++ct) {
    auto&& instr = instructions.at(ct);
    VLOG(5) << "Start running instruction-" << ct;
    auto run_start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat_times_; ++i) {
      instr->Run(&execution_args);
    }
#ifdef CINN_WITH_CUDA
    if (instr->target_ == common::DefaultNVGPUTarget()) {
      CUDA_CALL(cudaDeviceSynchronize());
    }
#endif
    auto time_span = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - run_start);
    auto cost_avg = static_cast<double>(time_span.count()) / repeat_times_;
    result.execution_cost += cost_avg;
  }

  auto time_span = std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::steady_clock::now() - t_start);
  result.elapsed_time = static_cast<double>(time_span.count());

  VLOG(4) << "A measurement done:repeat_times[" << repeat_times_
          << "]total_elapsed_time[" << result.elapsed_time
          << "]us,execution_cost[" << result.execution_cost << "]us";
  return result;
}

}  // namespace auto_schedule
}  // namespace cinn
