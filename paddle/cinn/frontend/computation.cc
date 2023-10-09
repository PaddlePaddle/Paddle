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

#include "paddle/cinn/frontend/computation.h"

#include "paddle/cinn/frontend/optimize.h"
#include "paddle/cinn/frontend/program_pass.h"
#include "paddle/cinn/hlir/framework/graph.h"
#include "paddle/cinn/hlir/framework/graph_compiler.h"
#include "paddle/cinn/hlir/framework/pass.h"
#include "paddle/cinn/hlir/framework/scope.h"

namespace cinn {
namespace frontend {

struct ComputationContext {
  Target target;
  void *stream;
  std::shared_ptr<hlir::framework::Graph> graph;
  std::shared_ptr<hlir::framework::Scope> scope;
  std::shared_ptr<hlir::framework::Program> program;
  std::shared_ptr<hlir::framework::GraphCompiler> graph_compiler;

  CinnComputation::CompileOptions compile_options;

  std::vector<hlir::framework::Tensor> inputs;
  std::vector<hlir::framework::Tensor> outputs;
  std::unordered_map<std::string, Variable> varmap;
  std::unordered_map<std::string, std::string> varmap_paddle2program;
};

std::shared_ptr<ComputationContext> CompileProgram(
    const Target &target,
    Program &program,  // NOLINT
    const std::vector<Variable> &outputs,
    std::shared_ptr<hlir::framework::Scope> scope,
    const CinnComputation::CompileOptions &options,
    void *stream) {
  std::shared_ptr<ComputationContext> ctx(new ComputationContext());
  ctx->stream = stream;
  ctx->target = target;
  ctx->compile_options = options;
  if (ctx->compile_options.use_decomposer) {
    ProgramPass::Apply(&program, {}, target, {"Decomposer"});
  }
  ctx->graph.reset(new hlir::framework::Graph(program, target));

  if (ctx->compile_options.use_default_passes) {
    hlir::framework::ApplyPass(ctx->graph.get(), "InferShape");

#ifndef CINN_WITH_CUDA
    if (target.arch == Target::Arch::X86) {
      hlir::framework::ApplyPass(ctx->graph.get(), "AlterLayout");
    }
#endif
    hlir::framework::ApplyPass(ctx->graph.get(), "ConstPropagate");
    hlir::framework::ApplyPasses(ctx->graph.get(), DefaultOpFusionPasses());
  }
  for (auto &pass_name : ctx->compile_options.passes) {
    hlir::framework::ApplyPass(ctx->graph.get(), pass_name);
  }

  ctx->scope = hlir::framework::BuildScope(target, ctx->graph, scope);

  std::unordered_set<std::string> fetch_var_ids;
  for (auto &out : outputs) {
    fetch_var_ids.insert(out->id);
  }

  ctx->compile_options.graph = ctx->graph;
  ctx->compile_options.scope = ctx->scope;
  ctx->compile_options.fetch_var_ids = fetch_var_ids;
  ctx->graph_compiler.reset(
      new hlir::framework::GraphCompiler(ctx->compile_options));
  ctx->program = ctx->graph_compiler->Build();
  if (ctx->compile_options.do_prerun) {
    ctx->program->PreRun();
  }

  for (auto &in_v : program.GetInputs()) {
    hlir::framework::Tensor t = ctx->scope->GetTensor(in_v->id);
    ctx->inputs.push_back(t);
  }
  for (auto &out_v : outputs) {
    hlir::framework::Tensor t = ctx->scope->GetTensor(out_v->id);
    ctx->outputs.push_back(t);
  }
  return ctx;
}

std::vector<std::string> CinnComputation::GetAllTensorNames() {
  std::vector<std::string> res;
  for (auto &v : context_->scope->var_names()) {
    res.push_back(std::string(v));
  }
  return res;
}

std::shared_ptr<CinnComputation> CinnComputation::CompilePaddleModel(
    const Target &target,
    const std::string &model_path,
    const std::vector<std::string> &input_names,
    const std::vector<hlir::framework::shape_t> &input_shapes,
    bool params_combined,
    const CompileOptions &options,
    void *stream) {
  CHECK(input_names.size() == input_shapes.size());
  auto scope = std::make_shared<hlir::framework::Scope>();
  std::unordered_map<std::string, std::vector<int>> input_shape_map;
  for (int idx = 0; idx < input_names.size(); ++idx) {
    input_shape_map[input_names[idx]] = input_shapes[idx];
  }
  auto loadedProgram = LoadPaddleProgram(
      model_path, scope.get(), input_shape_map, params_combined, target);
  auto &program = std::get<0>(loadedProgram);
  auto &varmap = std::get<1>(loadedProgram);
  auto &varmap_paddle2program = std::get<2>(loadedProgram);
  auto &fetch_names = std::get<3>(loadedProgram);

  // std::vector<Variable> input_vars;
  // for (int i = 0; i < input_names.size(); i++) {
  //   auto &name = input_names[i];
  //   auto &var  = varmap.at(name);
  //   var->shape = input_shapes[i];
  //   input_vars.push_back(var);
  // }
  // program->SetInputs({input_vars});
  // program->Validate();
  VLOG(3) << "program:\n" << *program;
  std::vector<Variable> output_vars;
  for (auto &name : fetch_names) {
    output_vars.push_back(varmap.at(name));
  }

  std::shared_ptr<ComputationContext> ctx =
      CompileProgram(target, *program, output_vars, scope, options, stream);
  for (auto &v : varmap) {
    ctx->varmap[v.first] = v.second;
  }
  for (auto &v : varmap_paddle2program) {
    ctx->varmap_paddle2program[v.first] = v.second;
  }

  auto computation = std::make_shared<CinnComputation>();
  computation->context_ = std::move(ctx);

  return computation;
}

std::shared_ptr<CinnComputation> CinnComputation::BuildAndCompile(
    const Target &target,
    NetBuilder &builder,
    const CompileOptions &options,
    const std::vector<Variable> &outputs,
    void *stream) {
  auto program = builder.Build();
  return Compile(target, program, options, outputs, stream);
}

std::shared_ptr<CinnComputation> CinnComputation::Compile(
    const Target &target,
    Program &program,
    const CompileOptions &options,
    const std::vector<Variable> &outputs,
    void *stream) {
  std::vector<Variable> output_vars = outputs;
  if (output_vars.empty()) {
    output_vars.push_back(program[program.size() - 1].GetOutput(0));
  }

  std::shared_ptr<ComputationContext> ctx =
      CompileProgram(target, program, output_vars, nullptr, options, stream);

  auto computation = std::make_shared<CinnComputation>();
  computation->context_ = std::move(ctx);

  return computation;
}

void CinnComputation::SetTensorData(const std::string &tname,
                                    void *data,
                                    size_t size) {
  hlir::framework::Tensor t = GetTensor(tname);
  SetTensorData(t, data, size);
}

void CinnComputation::SetTensorData(hlir::framework::Tensor &t,
                                    void *data,
                                    size_t size) {
  void *tdata = t->mutable_data(context_->target, t->type());
  CHECK_EQ(size, t->shape().numel() * t->type().bytes());
  if (context_->target.arch == Target::Arch::NVGPU) {
#ifdef CINN_WITH_CUDA
    CUDA_CALL(cudaMemcpy(tdata, data, size, cudaMemcpyHostToDevice));
#else
    CINN_NOT_IMPLEMENTED
#endif
  } else if (context_->target.arch == Target::Arch::X86) {
    memcpy(tdata, data, size);
  } else {
    CINN_NOT_IMPLEMENTED
  }
}
void CinnComputation::GetTensorData(hlir::framework::Tensor &t,
                                    void *data,
                                    size_t size) {
  void *tdata = t->mutable_data(context_->target, t->type());
  CHECK_EQ(size, t->shape().numel() * t->type().bytes());
  if (context_->target.arch == Target::Arch::NVGPU) {
#ifdef CINN_WITH_CUDA
    CUDA_CALL(cudaMemcpy(data, tdata, size, cudaMemcpyDeviceToHost));
#else
    CINN_NOT_IMPLEMENTED
#endif
  } else if (context_->target.arch == Target::Arch::X86) {
    memcpy(data, tdata, size);
  } else {
    CINN_NOT_IMPLEMENTED
  }
}

void CinnComputation::GetTensorData(const std::string &tname,
                                    void *data,
                                    size_t size) {
  hlir::framework::Tensor t = GetTensor(tname);
  GetTensorData(t, data, size);
}

std::vector<hlir::framework::Tensor> CinnComputation::GetInputTensors() {
  return context_->inputs;
}

std::vector<hlir::framework::Tensor> CinnComputation::GetOutputTensors() {
  return context_->outputs;
}

hlir::framework::Tensor CinnComputation::GetTensor(const std::string &tname) {
  if (context_->scope->FindVar(tname)) {
    return context_->scope->GetTensor(tname);
  }
  auto it = context_->varmap_paddle2program.find(tname);
  if (it == context_->varmap_paddle2program.end()) {
    LOG(FATAL) << "No variable called [" << tname
               << "] found in computation\nThe existing vars: "
               << utils::Join(context_->scope->var_names(), ", ");
  }
  return context_->scope->GetTensor(it->second);
}

void CinnComputation::Execute(
    const std::map<std::string, cinn_pod_value_t> *name2podargs) {
  context_->program->Execute(name2podargs, context_->stream);
}

}  // namespace frontend
}  // namespace cinn
