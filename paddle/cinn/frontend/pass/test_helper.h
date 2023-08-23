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

#pragma once

#include <gtest/gtest.h>

#include <random>

#include "paddle/cinn/frontend/net_builder.h"
#include "paddle/cinn/frontend/optimize.h"
#include "paddle/cinn/frontend/pass/use_program_pass.h"
#include "paddle/cinn/frontend/program_pass.h"
#include "paddle/cinn/hlir/framework/graph_compiler.h"
#include "paddle/cinn/hlir/framework/graph_compiler_util.h"
#include "paddle/cinn/hlir/framework/pass.h"
#include "paddle/cinn/hlir/pass/use_pass.h"

namespace cinn::frontend {

template <typename T>
std::vector<T> GeneratedRandomVector(size_t numel) {
  std::vector<T> data(numel);

  std::random_device seed;
  std::default_random_engine engine(seed());
  std::uniform_real_distribution<float> dist(0.f, 10.f);
  for (size_t i = 0; i < numel; i++) {
    data[i] = static_cast<T>(dist(engine));  // All random data
  }
  return data;
}

template <typename T>
void CopyFromVector(const std::vector<T>& src,
                    hlir::framework::Tensor tensor,
                    Target target) {
  size_t numel = tensor->shape().numel();
  auto* dst = tensor->mutable_data<T>(target);

#ifdef CINN_WITH_CUDA
  cudaMemcpy(dst, src.data(), numel * sizeof(T), cudaMemcpyHostToDevice);
#else
  std::copy(src.begin(), src.end(), dst);
#endif
}

template <typename T>
std::vector<T> CopyToVector(const hlir::framework::Tensor tensor) {
  size_t numel = tensor->shape().numel();
  auto* src = tensor->data<T>();

  std::vector<T> dst(numel);
#ifdef CINN_WITH_CUDA
  cudaMemcpy(dst.data(), src, numel * sizeof(T), cudaMemcpyDeviceToHost);
#else
  for (size_t i = 0; i < numel; ++i) {
    dst[i] = src[i];
  }
#endif
  return dst;
}

class PassTest {
 public:
  PassTest() { target_ = common::DefaultTarget(); }

  int RunAndCheck(NetBuilder* builder,
                  const std::vector<std::string>& program_passes,
                  const std::vector<std::string>& input_names,
                  const std::vector<std::string>& output_names) {
    auto program = builder->Build();
    CHECK(IsValid(program)) << "The origin program is not valid.";
    int origin_program_size = program.size();
    LOG(INFO) << "Run origin program";
    std::unordered_map<std::string, std::vector<float>> origin_outputs =
        Execute(program, input_names, output_names);

    std::unordered_set<std::string> fetch_var_ids(output_names.begin(),
                                                  output_names.end());
    ProgramPass::Apply(&program, fetch_var_ids, target_, program_passes);
    int optimized_program_size = program.size();
    CHECK(IsValid(program)) << "The optimized program is not valid.";
    LOG(INFO) << "Run optimized program";
    std::unordered_map<std::string, std::vector<float>> optimized_outputs =
        Execute(program, input_names, output_names);

    for (auto name : output_names) {
      LOG(INFO) << "Check output name=" << name;
      CHECK(origin_outputs.count(name));
      CHECK(optimized_outputs.count(name));
      CheckOutput(optimized_outputs[name], origin_outputs[name]);
    }
    return origin_program_size - optimized_program_size;
  }

 protected:
  std::unordered_map<std::string, std::vector<float>> Execute(
      const Program& program,
      const std::vector<std::string>& input_names,
      const std::vector<std::string>& output_names) {
    LOG(INFO) << program;
    std::unordered_set<std::string> fetch_var_ids(output_names.begin(),
                                                  output_names.end());
    auto graph = std::make_shared<hlir::framework::Graph>(
        program, fetch_var_ids, target_);
    hlir::framework::ApplyPasses(graph.get(), DefaultOpFusionPasses());

    auto scope = hlir::framework::BuildScope(target_, graph);
    hlir::framework::CompilationContext context(graph, scope, target_);
    context.with_instantiate_variables = true;
    hlir::framework::GraphCompiler gc(context);
    auto runtime_program = std::move(gc.Build());

    for (auto& name : input_names) {
      SetInputTensor(name, scope);
    }
    runtime_program->Execute();

    std::unordered_map<std::string, std::vector<float>> outputs;
    for (auto& name : output_names) {
      auto tensor = scope->GetTensor(name);
      std::vector<float> vec = CopyToVector<float>(tensor);
      outputs.emplace(name, vec);
    }
    return outputs;
  }

  void SetInputTensor(const std::string& name,
                      std::shared_ptr<hlir::framework::Scope> scope) {
    scope->Var<hlir::framework::Tensor>(name);
    auto tensor = scope->GetTensor(name);

    if (!inputs_.count(name)) {
      std::vector<float> vec =
          GeneratedRandomVector<float>(tensor->shape().numel());
      inputs_.emplace(name, vec);
    }
    auto iter = inputs_.find(name);
    CopyFromVector<float>(iter->second, tensor, target_);
  }

  void CheckOutput(const std::vector<float>& actual,
                   const std::vector<float>& expect) {
    CHECK_EQ(actual.size(), expect.size());
    for (size_t i = 0; i < expect.size(); ++i) {
      ASSERT_FLOAT_EQ(actual[i], expect[i]);
    }
  }

  bool IsValid(const Program& program) {
    std::unordered_set<std::string> inputs;
    for (auto& var : program.GetInputs()) {
      inputs.insert(var->id);
    }

    std::unordered_set<std::string> outputs;
    for (int i = 0; i < program.size(); ++i) {
      const auto& instr = program[i];
      for (auto& var : instr->outputs) {
        outputs.insert(var->id);
      }
    }

    bool valid = true;
    for (int i = 0; i < program.size(); ++i) {
      const auto& instr = program[i];
      // The inputs should be feeded, or other instructions' output.
      for (auto& var : instr->inputs) {
        if (!inputs.count(var->id) && !outputs.count(var->id)) {
          LOG(INFO) << "The input " << var->id << " of " << i
                    << "-th instrution (" << instr
                    << ") is not the output of any other instructions.";
          valid = false;
        }
      }
    }

    return valid;
  }

  Target target_;
  std::unordered_map<std::string, std::vector<float>> inputs_;
};

}  // namespace cinn::frontend
