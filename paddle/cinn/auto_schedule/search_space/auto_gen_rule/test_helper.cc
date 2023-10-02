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

#include "paddle/cinn/auto_schedule/search_space/auto_gen_rule/test_helper.h"

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <memory.h>
#include <stdlib.h>

#include "paddle/cinn/auto_schedule/analysis/analyze_ir.h"
#include "paddle/cinn/backends/codegen_cuda_dev.h"
#include "paddle/cinn/cinn.h"
#include "paddle/cinn/frontend/optimize.h"
#include "paddle/cinn/hlir/framework/instruction.h"
#include "paddle/cinn/hlir/framework/op.h"
#include "paddle/cinn/hlir/framework/op_lowering.h"
#include "paddle/cinn/hlir/framework/pass.h"
#include "paddle/cinn/hlir/framework/tensor.h"
#include "paddle/cinn/optim/transform_gpu_forloop.h"
#ifdef CINN_WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace cinn {
namespace auto_schedule {

using ::cinn::hlir::framework::Instruction;
using ::cinn::hlir::framework::Scope;
using ::cinn::hlir::framework::Shape;
using ::cinn::hlir::framework::Tensor;

void TestAutoGenRuleBase::Initialize(const common::Target& target) {
  target_ = target;
  backend_compier_ = backends::Compiler::Create(target);
}

ir::IRSchedule TestAutoGenRuleBase::MakeIRSchedule(
    const frontend::Program& test_program,
    utils::LinearRandomEngine::StateType rand_seed,
    bool apply_manual_schedule) {
  Context::Global().ResetNameId();

  auto graph = std::make_shared<hlir::framework::Graph>(test_program, target_);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");
  LOG_IF(WARNING, graph->fusion_groups.size() > 1)
      << "Test Graph has more than 1 group";
  auto& dtype_dict =
      graph->GetMutableAttrs<absl::flat_hash_map<std::string, common::Type>>(
          "inferdtype");
  auto& shape_dict = graph->GetMutableAttrs<
      absl::flat_hash_map<std::string, hlir::framework::shape_t>>("infershape");
  auto op_lowerer =
      hlir::framework::CreateOpLowerer(dtype_dict, shape_dict, target_);

  lowered_funcs_ =
      op_lowerer.Lower(graph->fusion_groups.front(),
                       /*apply_op_schedule = */ apply_manual_schedule,
                       /*apply_group_schedule = */ apply_manual_schedule);
  CHECK(!lowered_funcs_.empty()) << "lowered_funcs_ is empty";

  std::vector<Expr> bodys;
  for (auto&& func : lowered_funcs_) {
    bodys.emplace_back(func->body);
  }
  return ir::IRSchedule(ir::ModuleExpr({std::move(bodys)}), rand_seed);
}

std::string TestAutoGenRuleBase::GetIR(const ir::IRSchedule& schedule) {
  const auto& exprs = schedule.GetModule().GetExprs();
  std::stringstream module_stream;
  for (auto i = 0; i < exprs.size(); ++i) {
    module_stream << "Expr " << i << " {\n"
                  << exprs.at(i) << "\n}  // end Expr " << i << "\n";
  }
  return module_stream.str();
}

ir::Module TestAutoGenRuleBase::BuildIRModule(const ir::IRSchedule& schedule) {
  auto&& updated_bodys = schedule.GetModule().GetExprs();
  CHECK_EQ(lowered_funcs_.size(), updated_bodys.size())
      << "associated exprs size not equal";

  ir::Module::Builder builder("test_bulder", this->target_);
  for (int i = 0; i < lowered_funcs_.size(); ++i) {
    ir::Expr func_body = updated_bodys.at(i);
    const ir::LoweredFunc& ori_func = lowered_funcs_.at(i);
    auto&& new_func = UpdateFuncWithNewBody(target_, ori_func, func_body);
    builder.AddFunction(new_func);
  }

  return builder.Build();
}

std::string TestAutoGenRuleBase::GenSourceCode(const ir::Module& ir_module) {
  std::unique_ptr<backends::CodeGenC> codegen;
#ifdef CINN_WITH_CUDA
  if (target_ == common::DefaultNVGPUTarget()) {
    codegen = std::make_unique<backends::CodeGenCUDA_Dev>(this->target_);
  } else {
    codegen = std::make_unique<backends::CodeGenCX86>(
        this->target_, CodeGenCX86::Feature::AVX512);
  }
#else
  codegen = std::make_unique<backends::CodeGenCX86>(
      this->target_, CodeGenCX86::Feature::AVX512);
#endif
  codegen->SetInlineBuiltinCodes(false);
  return codegen->Compile(ir_module, CodeGenC::OutputKind::CImpl);
}

raw_func_type TestAutoGenRuleBase::GenExecutableKernel(
    const ir::Module& ir_module) {
  auto&& func_name = lowered_funcs_.front()->name;
  // Compile to machine code
  backend_compier_->Build(ir_module);
  auto test_func_ptr = reinterpret_cast<void (*)(void**, int32_t)>(
      backend_compier_->Lookup(func_name));
  return test_func_ptr;
}

void MemoryCopy(const float* src, float* dst, int numel, std::string type) {
#ifdef CINN_WITH_CUDA
  if (type == "DeviceToHost") {
    cudaMemcpy(dst, src, numel * sizeof(float), cudaMemcpyDeviceToHost);
    return;
  } else if (type == "HostToDevice") {
    cudaMemcpy(dst, src, numel * sizeof(float), cudaMemcpyHostToDevice);
    return;
  }
#endif
  if (type == "HostToHost") {
    for (size_t i = 0; i < numel; ++i) {
      dst[i] = src[i];
    }
  } else {
    LOG(FATAL) << "Unknown memory copy type";
  }
}

void AddDataToScope(Scope* scope,
                    const common::Target& target,
                    float* data_ptr,
                    std::string name,
                    const std::vector<int>& shape) {
  auto* var = scope->Var<Tensor>(name);
  auto& tensor = absl::get<Tensor>(*var);
  CHECK(shape.size()) << "The size of shape can not be 0.";
  Shape cinn_shape(shape);
  tensor->Resize(cinn_shape);
  auto* tgt_data_ptr = tensor->mutable_data<float>(target);
  std::string mem_cpy_type =
      target == common::DefaultNVGPUTarget() ? "DeviceToHost" : "HostToHost";
  MemoryCopy(data_ptr, tgt_data_ptr, cinn_shape.numel(), mem_cpy_type);
}

void CheckResult(raw_func_type test_func,
                 raw_func_type expected_func,
                 const std::vector<std::string>& input_names,
                 const std::vector<std::string>& output_names,
                 const std::vector<std::vector<int>>& input_shapes,
                 const std::vector<std::vector<int>>& output_shapes,
                 const common::Target& target) {
  CHECK(input_names.size()) << "The number of inputs must be greater than 0.";
  CHECK(output_names.size()) << "The number of outputs must be greater than 0.";
  CHECK_EQ(input_names.size(), input_shapes.size())
      << "The quantity of input_names and input_shapes must be equal.";
  CHECK_EQ(output_names.size(), output_shapes.size())
      << "The quantity of output_names and output_shapes must be equal.";

  // Initialize data
  std::vector<float*> input_data_ptrs(input_names.size());
  for (int i = 0; i < input_shapes.size(); ++i) {
    int input_data_numel = std::accumulate(
        input_shapes[i].begin(), input_shapes[i].end(), 1, [](int a, int b) {
          return a * b;
        });
    input_data_ptrs[i] =
        reinterpret_cast<float*>(malloc(input_data_numel * sizeof(float)));
    for (int j = 0; j < input_data_numel; ++j) {
      input_data_ptrs[i][j] = (rand() * 1.f) / RAND_MAX;  // NOLINT
    }
  }
  std::vector<float*> test_output_data_ptrs(output_names.size());
  std::vector<float*> expected_output_data_ptrs(output_names.size());
  std::vector<int> output_data_numels(output_shapes.size());
  for (int i = 0; i < output_shapes.size(); ++i) {
    output_data_numels[i] = std::accumulate(
        output_shapes[i].begin(), output_shapes[i].end(), 1, [](int a, int b) {
          return a * b;
        });
    test_output_data_ptrs[i] =
        reinterpret_cast<float*>(malloc(output_data_numels[i] * sizeof(float)));
    memset(test_output_data_ptrs[i], 0, output_data_numels[i] * sizeof(float));
    expected_output_data_ptrs[i] =
        reinterpret_cast<float*>(malloc(output_data_numels[i] * sizeof(float)));
    memset(
        expected_output_data_ptrs[i], 0, output_data_numels[i] * sizeof(float));
  }

  auto launch_kernel_fn = [&](raw_func_type& raw_func,
                              std::vector<float*>& output_data_ptrs) {
    // Initialize scope
    Scope scope;
    // Initialize input data in scope.
    for (int i = 0; i < input_names.size(); ++i) {
      AddDataToScope(
          &scope, target, input_data_ptrs[i], input_names[i], input_shapes[i]);
    }
    // Initialize output data in scope.
    for (int i = 0; i < output_names.size(); ++i) {
      AddDataToScope(&scope,
                     target,
                     output_data_ptrs[i],
                     output_names[i],
                     output_shapes[i]);
    }

    // Create Instruction and run
    Instruction instr(target, &scope, input_names, output_names);
    CHECK(raw_func) << "The raw_func can not be nullptr.";
    instr.SetLoweredFunc(reinterpret_cast<void*>(raw_func));
    // should call Finalize explicitly before Run
    instr.Finalize();
    instr.Run();

    // data
    for (int i = 0; i < output_names.size(); ++i) {
      const float* result_ptr = scope.GetTensor(output_names[i])->data<float>();
      std::string mem_cpy_type = target == common::DefaultNVGPUTarget()
                                     ? "DeviceToHost"
                                     : "HostToHost";
      MemoryCopy(
          result_ptr, output_data_ptrs[i], output_data_numels[i], mem_cpy_type);
    }
  };

  // launch and execute test and expected kernel separately
  launch_kernel_fn(test_func, test_output_data_ptrs);
  launch_kernel_fn(expected_func, expected_output_data_ptrs);

  // Check result
  for (int i = 0; i < output_shapes.size(); ++i) {
    for (int j = 0; j < output_data_numels[i]; ++j) {
      ASSERT_NEAR(
          test_output_data_ptrs[i][j], expected_output_data_ptrs[i][j], 1e-4);
    }
  }

  // Free memory
  for (auto ptr : input_data_ptrs) {
    free(ptr);
  }
  for (auto ptr : test_output_data_ptrs) {
    free(ptr);
  }
  for (auto ptr : expected_output_data_ptrs) {
    free(ptr);
  }
}

}  // namespace auto_schedule
}  // namespace cinn
