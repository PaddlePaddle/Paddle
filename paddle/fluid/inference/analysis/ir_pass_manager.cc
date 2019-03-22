// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/inference/analysis/ir_pass_manager.h"
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/inference/analysis/argument.h"
#include "paddle/fluid/inference/analysis/ir_passes/subgraph_detector.h"
#include "paddle/fluid/string/pretty_log.h"

namespace paddle {
namespace inference {
namespace analysis {
using string::PrettyLogEndl;
using string::PrettyLog;
using string::Style;

IRPassManager::IRPassManager(Argument *argument) {
  ARGUMENT_CHECK_FIELD(argument, main_program);
  graph_ = std::unique_ptr<Graph>(new Graph(argument->main_program()));
  if (argument->Has("scope")) {
    graph_->Set(framework::ir::kParamScopeAttr,
                new framework::Scope *(
                    const_cast<framework::Scope *>(&argument->scope())));
  }

  ARGUMENT_CHECK_FIELD(argument, ir_analysis_passes);
  CreatePasses(argument, argument->ir_analysis_passes());
}

void IRPassManager::CreatePasses(Argument *argument,
                                 const std::vector<std::string> &passes) {
  std::string pre_pass;
  int pass_num = 0;
  for (const std::string &pass_name : passes) {
    auto pass = framework::ir::PassRegistry::Instance().Get(pass_name);

    if (pass_name == "graph_viz_pass") {
      std::string dot_file_path = std::to_string(pass_num) + "_ir_" +
                                  (pre_pass.empty() ? "origin" : pre_pass) +
                                  ".dot";
      pass->Set("graph_viz_path", new std::string(std::move(dot_file_path)));
      pass_num++;
    } else if (pass_name == "mkldnn_placement_pass") {
      pass->Set("mkldnn_enabled_op_types",
                new std::unordered_set<std::string>(
                    argument->mkldnn_enabled_op_types()));
    } else if (pass_name == "cpu_quantize_placement_pass") {
      pass->Set("quantize_enabled_op_types",
                new std::unordered_set<std::string>(
                    argument->quantize_enabled_op_types()));
      pass->Set(
          "quantize_excluded_op_ids",
          new std::unordered_set<int>(argument->quantize_excluded_op_ids()));
    } else if (pass_name == "cpu_quantize_pass") {
      pass->Set("quant_var_scales",
                new VarQuantScale(argument->quant_var_scales()));
    }

    if (pass_name == "anakin_subgraph_pass") {
      pass->Set("program",
                new framework::ProgramDesc *(&argument->main_program()));
      pass->Set("gpu_device_id", new int(argument->gpu_device_id()));
      pass->Set("model_from_memory", new bool(argument->model_from_memory()));
      pass->Set("engine_opt_info", new std::map<std::string, std::string>(
                                       argument->engine_opt_info()));
      pass->Set("predictor_id", new int(argument->predictor_id()));
      pass->Set("max_input_shape", new std::map<std::string, std::vector<int>>(
                                       argument->anakin_max_input_shape()));
      pass->Set("max_batch_size", new int(argument->anakin_max_batch_size()));
    }

    if (pass_name == "tensorrt_subgraph_pass") {
      pass->Set("workspace_size", new int(argument->tensorrt_workspace_size()));
      pass->Set("max_batch_size", new int(argument->tensorrt_max_batch_size()));
      pass->Set("min_subgraph_size",
                new int(argument->tensorrt_min_subgraph_size()));
      pass->Set("program",
                new framework::ProgramDesc *(&argument->main_program()));

      bool enable_int8 = argument->tensorrt_precision_mode() ==
                         AnalysisConfig::Precision::kInt8;

      pass->Set("enable_int8", new bool(enable_int8));

      bool use_static_engine = argument->tensorrt_use_static_engine();
      bool model_from_memory = argument->model_from_memory();
      bool int8_valid = !(model_from_memory && enable_int8);
      PADDLE_ENFORCE(int8_valid,
                     "TRT INT8 Now don't support model load from memory.");

      if ((!model_from_memory && use_static_engine) || enable_int8) {
        std::string model_opt_cache_dir =
            argument->Has("model_dir")
                ? argument->model_dir()
                : GetDirRoot(argument->model_program_path());
        pass->Set(
            "model_opt_cache_dir",
            new std::string(GetOrCreateModelOptCacheDir(model_opt_cache_dir)));
      }
      pass->Set("gpu_device_id", new int(argument->gpu_device_id()));
      pass->Set("use_static_engine", new bool(use_static_engine));
      pass->Set("model_from_memory", new bool(argument->model_from_memory()));
      pass->Set("engine_opt_info", new std::map<std::string, std::string>(
                                       argument->engine_opt_info()));
    }

    pre_pass = pass_name;

    passes_.emplace_back(std::move(pass));
  }
}

std::unique_ptr<Graph> IRPassManager::Apply(std::unique_ptr<Graph> graph) {
  if (passes_.empty()) {
    return graph;
  }
  PADDLE_ENFORCE(graph.get());
  // Apply all the passes
  for (const auto &pass : passes_) {
    if (pass->Type() != "graph_viz_pass") {
      PrettyLogEndl(Style::H2(), "--- Running IR pass [%s]", pass->Type());
    }
    graph = pass->Apply(std::move(graph));
  }
  return graph;
}

framework::proto::ProgramDesc IRPassManager::AcquireProgram(
    std::unique_ptr<Graph> *graph, ProgramDesc *program) const {
  auto pass =
      framework::ir::PassRegistry::Instance().Get("graph_to_program_pass");

  // Direct using ProgramDesc desc(argument->main_program()) may cause
  // incomplete copies of information.
  ProgramDesc desc;
  desc.CopyFrom(*program->Proto());
  pass->SetNotOwned("program", &desc);
  auto *the_graph = graph->release();
  *graph = pass->Apply(std::unique_ptr<Graph>(the_graph));
  return *desc.Proto();
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
