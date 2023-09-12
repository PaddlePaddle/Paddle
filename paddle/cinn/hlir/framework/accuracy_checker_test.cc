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

#include "paddle/cinn/hlir/framework/accuracy_checker.h"

#include <gtest/gtest.h>

#include <random>
#ifdef CINN_WITH_CUDA
#include <cuda_runtime.h>
#endif

#include "paddle/cinn/backends/llvm/simple_jit.h"
#include "paddle/cinn/hlir/framework/instruction.h"
#include "paddle/cinn/hlir/framework/op_strategy.h"

PD_DECLARE_string(cinn_self_check_accuracy);

namespace cinn {
namespace hlir {
namespace framework {

void GenerateRandomData(float* data, size_t numel, bool generate_nan) {
  std::random_device seed;
  std::default_random_engine engine(seed());
  std::uniform_real_distribution<float> dist(-100.f, 100.f);
  for (size_t i = 0; i < numel; i++) {
    float v = dist(engine);
    data[i] = generate_nan ? sqrt(v) : v;
  }
}

void SetRandomTensor(Tensor tensor, Target target, bool generate_nan) {
  size_t numel = tensor->shape().numel();
  float* dst = tensor->mutable_data<float>(target);

  std::vector<float> random_nan_vec(numel);
  GenerateRandomData(random_nan_vec.data(), numel, generate_nan);

#ifdef CINN_WITH_CUDA
  if (target == common::DefaultNVGPUTarget()) {
    cudaMemcpy(dst,
               random_nan_vec.data(),
               numel * sizeof(float),
               cudaMemcpyHostToDevice);
  }
#endif
  if (target == common::DefaultHostTarget()) {
    std::copy(random_nan_vec.begin(), random_nan_vec.end(), dst);
  }
}

TEST(AccuracyChecker, tensor) {
  Target target = common::DefaultTarget();
  Scope scope;
  scope.Var<Tensor>("x");
  auto out = scope.GetTensor("x");
  out->Resize(Shape({16, 16}));
  SetRandomTensor(out, target, true);

  AccuracyChecker checker(target, &scope);
  std::string result_str = checker("x");
  LOG(INFO) << result_str;
}

std::unique_ptr<backends::SimpleJIT> GetLoweredFunc(Target target) {
  Expr m(16);
  Expr n(16);

  lang::Placeholder<float> x("x", {m, n});

  auto y = Compute(
      {m, n},
      [=](Expr i, Expr j) { return lang::CallExtern("sqrt", {x(i, j)}); },
      "y");

  auto stages = CreateStages({y});
  auto fn = Lower("fn_sqrt", stages, {x, y});

  ir::Module::Builder builder("some_module", target);
  builder.AddFunction(fn);

  auto jit = backends::SimpleJIT::Create();
  jit->Link(builder.Build());
  return std::move(jit);
}

void InstantiateScope(Scope* scope, Target target) {
  for (auto& name : std::vector<std::string>({"x", "y"})) {
    scope->Var<Tensor>(name);
    auto x = scope->GetTensor(name);
    x->Resize(Shape({16, 16}));
    SetRandomTensor(x, target, false);
  }
}

TEST(AccuracyChecker, instruction) {
  Target target = common::DefaultHostTarget();
  Scope scope;
  InstantiateScope(&scope, target);

  auto jit = GetLoweredFunc(target);
  auto fn_ptr = jit->Lookup("fn_sqrt");
  CHECK(fn_ptr);

  FLAGS_cinn_self_check_accuracy = "true";
  Instruction instr(target, &scope, {"x"}, {"y"});
  instr.SetLoweredFunc(reinterpret_cast<void*>(fn_ptr), "fn_sqrt");
  // should call Finalize explicitly before Run
  instr.Finalize();

  instr.Run();
  FLAGS_cinn_self_check_accuracy = "";
}

void InitName2PodArgs(Target target,
                      std::vector<cinn_buffer_t>* args_buffer,
                      std::map<std::string, cinn_pod_value_t>* name2podargs) {
  auto* default_memory_mng =
      MemoryManager::Global().RetrieveSafely(target.arch);

  int count = 0;
  const auto& shape = Shape({16, 16});
  size_t numel = shape.numel();
  for (const auto& name : std::vector<std::string>({"x", "y"})) {
    auto* buffer = &args_buffer->at(count++);
    buffer->type = cinn_float32_t();
    buffer->resize(
        reinterpret_cast<const cinn_dimension_t*>(shape.data().data()),
        shape.size());
    buffer->memory = reinterpret_cast<uint8_t*>(
        default_memory_mng->malloc(numel * sizeof(float)));
    float* data = reinterpret_cast<float*>(buffer->memory);
    GenerateRandomData(data, numel, false);
    name2podargs->emplace(name, buffer);
  }
}

TEST(AccuracyChecker, instruction_podargs) {
  Target target = common::DefaultHostTarget();
  std::vector<cinn_buffer_t> args_buffer(2);
  std::map<std::string, cinn_pod_value_t> name2podargs;
  InitName2PodArgs(target, &args_buffer, &name2podargs);

  auto jit = GetLoweredFunc(target);
  auto fn_ptr = jit->Lookup("fn_sqrt");
  CHECK(fn_ptr);

  FLAGS_cinn_self_check_accuracy = "true";
  Instruction instr(target, nullptr, {"x"}, {"y"});
  instr.SetLoweredFunc(reinterpret_cast<void*>(fn_ptr), "fn_sqrt");
  instr.Finalize();

  instr.Run(&name2podargs);
  FLAGS_cinn_self_check_accuracy = "";
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
