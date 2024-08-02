// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/cinn/operator_fusion/backend/pattern.h"
#include "paddle/cinn/operator_fusion/backend/pattern_fuser.h"
#include "paddle/cinn/operator_fusion/frontend/pattern.h"
#include "paddle/cinn/operator_fusion/frontend/pattern_fuser.h"
#include "paddle/cinn/operator_fusion/pattern_graph.h"
#include "paddle/cinn/operator_fusion/policy/general_topo_policy.h"
#include "paddle/cinn/operator_fusion/policy/relative_judge_policy.h"
#include "paddle/cinn/operator_fusion/policy/shardable_axes_policy.h"

namespace cinn::fusion {

template <typename T>
inline std::vector<fusion::PatternNodePtr<T>> ClusterOps(
    const std::vector<fusion::PatternContent<T>>& contents,
    const std::vector<pir::Value>& output_values) {
  std::function<pir::Operation*(fusion::PatternContent<T>)> func =
      [](const fusion::PatternContent<T>& content) { return content.op; };
  const auto& origin_ops = fusion::MapVector(contents, func);
  PADDLE_ENFORCE_GT(origin_ops.size(),
                    0,
                    phi::errors::InvalidArgument(
                        "The size of origin_ops should be greater than 0. "));
  VLOG(4) << "Start Cluster Ops!";
  VLOG(4) << "Input Group with size " << origin_ops.size() << " :\n"
          << fusion::OpsDebugStr(origin_ops);

  std::vector<pir::Value> outputs = output_values;
  const auto& ops = [&] {
    std::vector<pir::Operation*> ops;
    for (const auto& content : contents) {
      if (content.op->name() == "cf.yield") {  // just skip cf.yield.
        for (auto& operand : content.op->operands()) {
          outputs.push_back(operand.source());
        }
        continue;
      }
      ops.emplace_back(content.op);
    }
    return ops;
  }();

  const auto& content_without_yield =
      FilterVector(contents, [](const fusion::PatternContent<T>& content) {
        return content.op->name() != "cf.yield";
      });

  pir::Program* program = origin_ops.at(0)->GetParentProgram();

  auto* shape_analysis = &pir::ShapeAnalysisManager::Instance().Get(program);

  VLOG(4) << "Start Create Policies and PolicyManager!";
  const auto& relative_judge_policy =
      std::make_shared<fusion::RelativeJudgePolicy<T>>(ops, shape_analysis);

  const auto& general_topo_policy =
      std::make_shared<fusion::GeneralTopoPolicy<T>>();

  auto policy_manager =
      fusion::PolicyManager<T>({relative_judge_policy, general_topo_policy});

  auto topo_manager = fusion::PolicyManager<T>({general_topo_policy});

  VLOG(4) << "Start Create PatternGraph";
  fusion::PatternGraph<T> graph(
      content_without_yield, outputs, policy_manager, topo_manager);
  auto result = graph.ClusterOps();

  VLOG(4) << "End Cluster Ops! result size:" << result.size();
  for (const auto& node : result) {
    VLOG(4) << "\n"
            << node->DebugStr() << "\n"
            << fusion::StmtPatternDebugStr(node->stmt_pattern());
  }

  return result;
}

}  // namespace cinn::fusion
