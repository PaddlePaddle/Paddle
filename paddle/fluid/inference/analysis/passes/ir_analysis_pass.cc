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

#include "paddle/fluid/inference/analysis/passes/ir_analysis_pass.h"
#include "paddle/fluid/framework/ir/graph_helper.h"

#include <memory>
#include <utility>

#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/inference/analysis/ir_pass_manager.h"

#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/framework/ir/mkldnn/mkldnn_pass_util.h"
#endif

namespace paddle {
namespace inference {
namespace analysis {

void IrAnalysisPass::RunImpl(Argument* argument) {
  ARGUMENT_CHECK_FIELD(argument, ir_analysis_passes);
  ARGUMENT_CHECK_FIELD(argument, main_program);
  ARGUMENT_CHECK_FIELD(argument, scope);

  auto* the_graph = argument->ReleaseMainGraph();
  auto graph = std::unique_ptr<Graph>(the_graph);

  std::unordered_map<std::string, std::vector<float>> var_quant_scales{};
  if (argument->Has("scale_file_path")) {
    VLOG(5) << "Scale file path for quantized model: "
            << argument->scale_file_path();
    std::ifstream out_scale_file(argument->scale_file_path());
    std::string one_line;
    while (getline(out_scale_file, one_line)) {
      if (one_line.find(" ") != one_line.npos) {
        auto pos = one_line.find(" ");
        std::string pre_str = one_line.substr(0, pos);
        std::string pos_str = one_line.substr(pos);
        if (pre_str.size() && pos_str.size()) {
          std::string tensor_name = pre_str;
          float scale = std::stod(pos_str);
          if (std::isinf(scale) || std::isnan(scale)) {
            continue;
          }
          std::vector<float> scales = {scale};
          var_quant_scales[tensor_name] = scales;
        }
      }
    }
  }

  auto op_node_sorted = framework::ir::TopologyVarientSort(
      *graph, static_cast<framework::ir::SortKind>(0));
  for (auto* op_node : op_node_sorted) {
    auto op_desc = op_node->Op();
    for (auto iter : op_desc->Inputs()) {
      for (size_t i = 0; i < iter.second.size(); i++) {
        if (var_quant_scales.count(iter.second[i])) {
          op_desc->SetAttr(iter.first + "_" + std::to_string(i),
                           var_quant_scales[iter.second[i]][0]);
        }
      }
    }
    for (auto iter : op_desc->Outputs()) {
      for (size_t i = 0; i < iter.second.size(); i++) {
        if (var_quant_scales.count(iter.second[i])) {
          op_desc->SetAttr(iter.first + "_" + std::to_string(i),
                           var_quant_scales[iter.second[i]][0]);
        }
      }
    }
  }

  // Apply passes.
  IRPassManager the_ir_manager(argument);
  graph = the_ir_manager.Apply(std::move(graph));
  PADDLE_ENFORCE_GT(
      graph->Nodes().size(),
      0,
      platform::errors::PreconditionNotMet(
          "The graph nodes size should be greater than 0, but got 0"));
  argument->SetMainGraph(graph.release());
  CollectFusionStatis(argument);
}

void IrAnalysisPass::CollectFusionStatis(Argument* argument) {
  if (!argument->main_graph().Has(framework::ir::kFuseStatisAttr)) {
    LOG(INFO) << "argument has no fuse statis";
    return;
  }
  argument->SetFusionStatis(
      argument->main_graph().Get<Argument::fusion_statis_t>(
          framework::ir::kFuseStatisAttr));
}

std::string IrAnalysisPass::repr() const { return "ir-analysis-pass"; }

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
