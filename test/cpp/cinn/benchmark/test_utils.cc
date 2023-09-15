// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "test/cpp/cinn/benchmark/test_utils.h"

#include "paddle/cinn/backends/llvm/codegen_x86.h"
#include "paddle/cinn/common/cas.h"
#include "paddle/cinn/common/test_helper.h"
#include "paddle/cinn/hlir/framework/graph_compiler.h"
#include "paddle/cinn/hlir/framework/op.h"
#include "paddle/cinn/hlir/framework/op_strategy.h"
#include "paddle/cinn/optim/transform_gpu_forloop.h"
#include "paddle/cinn/runtime/flags.h"
#include "paddle/cinn/utils/timer.h"

namespace cinn {
namespace tests {
using ir::Tensor;
std::unique_ptr<backends::ExecutionEngine>
OpBenchmarkTester::CreateExecutionEngine(const cinn::ir::Module& module) {
  auto engine = backends::ExecutionEngine::Create({});
  engine->Link<backends::CodeGenX86>(module);
  return engine;
}

void OpBenchmarkTester::TestOp(const std::string& test_name,
                               const std::vector<Tensor>& input_tensors,
                               const hlir::framework::NodeAttr& attrs,
                               const std::vector<Type>& input_types,
                               const std::vector<Type>& out_types,
                               bool use_default_stragegy) {
  auto module =
      CreateCinnModule(input_tensors, attrs, out_types, use_default_stragegy);
  auto engine = CreateExecutionEngine(module);
  auto test_func_ptr =
      reinterpret_cast<void (*)(void**, int32_t)>(engine->Lookup(op_name_));
  input_types_ = input_types;
  out_types_ = out_types;
  CreateBuffer();
  LOG(INFO) << "Testing " << test_name;
  cinn::utils::Timer timer;
  // ignore first execution for lazy jit component
  timer.Start();
  test_func_ptr(reinterpret_cast<void**>(all_args_.data()), all_args_.size());
  double test_op_time = timer.Stop();
  LOG(INFO) << "kernel warmup run time: " << test_op_time << " ms";
  timer.Start();
  for (int i = 0; i < repeat_; i++) {
    test_func_ptr(reinterpret_cast<void**>(all_args_.data()), all_args_.size());
  }
  test_op_time = timer.Stop() / repeat_;
  LOG(INFO) << "repeat times: " << repeat_
            << ", kernel run time: " << test_op_time << " ms";
}

Module OpBenchmarkTester::CreateCinnModule(
    const std::vector<Tensor>& input_tensors,
    const hlir::framework::NodeAttr& attrs,
    const std::vector<Type>& out_types,
    bool use_default_stragegy) {
  std::vector<Tensor> outs;
  std::vector<Tensor> rets;
  poly::StageMap stages;
  CHECK(!out_types.empty());
  rets = input_tensors;
  Module::Builder builder("module_" + op_name_, target_);

  if (use_default_stragegy) {
    auto strategy =
        hlir::framework::Operator::GetAttrs<hlir::framework::StrategyFunction>(
            "CINNStrategy");
    auto op = hlir::framework::Operator::Get(op_name_);
    CHECK(op) << op_name_ << " isn't supported yet\n";
    auto impl = hlir::framework::OpStrategy::SelectImpl(
        strategy[op](attrs, input_tensors, out_types, input_shapes_, target_));

    std::string output_name = "out";
    std::vector<common::CINNValue> temp_inputs;
    std::vector<ir::Tensor> all_arg_tensors;
    std::vector<std::string> input_output_names;
    for (const auto& tensor : input_tensors) {
      temp_inputs.emplace_back(tensor);
      all_arg_tensors.push_back(tensor);
      input_output_names.push_back(tensor->name);
    }
    temp_inputs.emplace_back(output_name);
    common::CINNValuePack cinn_inputs = common::CINNValuePack{temp_inputs};
    input_output_names.push_back(output_name);

    // 1.Call Op's Compute function, using the default stages and LowerVec to
    // get IR tree.
    common::CINNValuePack C = impl->fcompute(cinn_inputs);

    // 2. Collect tensors and arguments
    // Add output tensors to all_arg_tensors
    for (int i = 0; i < C->size() - 1; i++) {
      ir::Expr temp = C[i];
      // checkout whether the tensor is with buffer.
      if (!temp.as_tensor_ref()->buffer.defined() ||
          target_ != common::DefaultNVGPUTarget()) {
        all_arg_tensors.push_back(temp.as_tensor_ref());
      }
    }

    stages = C.back();
    auto funcs = lang::LowerVec(
        op_name_, stages, all_arg_tensors, {}, {}, nullptr, target_, true);

    std::vector<common::CINNValue> schedule_inputs;
    for (int i = 0; i < C.size() - 1; ++i) {
      CHECK(C[i].is_tensor());
      schedule_inputs.push_back(common::CINNValue(C[i]));
    }
    for (auto& f : funcs) {
      schedule_inputs.push_back(common::CINNValue(f->body));
    }

    // 3. Call Op's Schedule function, optimizing the IR tree by new IR
    // schedule
    common::CINNValuePack expr_pack =
        impl->fschedule(common::CINNValuePack{schedule_inputs});

    // 4. Optimize the LoweredFunc
    std::vector<ir::LoweredFunc> res;
    for (int i = 0; i < expr_pack.size(); i++) {
#ifdef CINN_WITH_CUDA
      optim::OptimizeExprGPU(&(funcs[i]->body));
#endif
      if (funcs.size() > expr_pack.size()) {
        auto new_args = lang::GetArgs(funcs[i]->body, input_output_names);
        funcs[i]->args = new_args;
      }
      auto temp_buffers =
          lang::GetTempBuffers(all_arg_tensors, stages, funcs[i]->body);
      funcs[i]->temp_bufs = temp_buffers;
      funcs[i]->PrepareBufferCastExprs();
      res.push_back(funcs[i]);
    }
    for (int i = 0; i < res.size(); i++) {
      res[i] =
          optim::Optimize(Expr(funcs[i]), target_, false).as_lowered_func_ref();
    }

    for (auto func : res) {
      builder.AddFunction(func);

      for (const auto& arg : func->args) {
        std::vector<int> output_shape;
        if (arg.io == ir::Argument::IO::kOutput) {
          for (auto& shape_dim : arg.buffer_arg()->shape) {
            LOG(INFO) << shape_dim << ",";
            CHECK(shape_dim.is_constant());
            output_shape.push_back(static_cast<int>(shape_dim.get_constant()));
          }
          output_shapes_.push_back(output_shape);
          break;
        }
      }
    }
  } else {
    stages = CreateStages(input_tensors);
    outs = CreateSpecificStrategy(input_tensors, &stages);

    for (auto& out : outs) {
      stages->InsertLazily(out);
      rets.push_back(out);
      std::vector<Expr> output_shape_expr = out->domain_without_reduce_axis();
      std::vector<int> output_shape;
      for (auto& shape : output_shape_expr) {
        output_shape.push_back(shape.as_int32());
      }
      output_shapes_.push_back(output_shape);
    }
    auto func = Lower(op_name_, stages, rets);
    LOG(INFO) << "After Lower, func is: \n" << func;

    builder.AddFunction(func);
  }

  CodeGenC compiler(target_);
  Outputs outputs;
  outputs = outputs.c_header("./test_" + op_name_ + ".h")
                .c_source("./test_" + op_name_ + ".cc");
  compiler.Compile(builder.Build(), outputs);
  return builder.Build();
}

void OpBenchmarkTester::CreateBuffer() {
  std::vector<cinn_pod_value_t> args;
  for (size_t i = 0; i < input_shapes_.size(); i++) {
    auto* buffer = common::BufferBuilder(input_types_[i], input_shapes_[i])
                       .set_align(32)
                       .set_random()
                       .Build();
    cinn_pod_value_t arg(buffer);
    all_args_.push_back(arg);
  }
  CHECK(!output_shapes_.empty()) << "output shapes shouldn't be empty\n";
  CHECK_EQ(output_shapes_.size(), out_types_.size());
  for (size_t i = 0; i < output_shapes_.size(); i++) {
    if (out_types_[i].is_void()) continue;
    auto* buffer = common::BufferBuilder(out_types_[i], output_shapes_[i])
                       .set_align(32)
                       .set_zero()
                       .Build();
    CHECK(buffer);
    out_dims_ = buffer->num_elements();
    cinn_pod_value_t arg(buffer);
    all_args_.push_back(arg);
  }
}

}  // namespace tests
}  // namespace cinn
