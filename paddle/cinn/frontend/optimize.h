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

#pragma once

#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "paddle/cinn/common/target.h"
#include "paddle/cinn/frontend/syntax.h"
#include "paddle/cinn/hlir/framework/graph.h"

namespace cinn {
namespace frontend {

struct OptimizeOptions {
  std::vector<std::string> program_passes;
  std::vector<std::string> graph_passes;
};

OptimizeOptions DefaultTrainingOptimizeOptions();

std::vector<std::string> DefaultOpFusionPasses();

std::shared_ptr<hlir::framework::Graph> Optimize(
    frontend::Program* program,
    const std::unordered_set<std::string>& fetch_ids,
    common::Target target,
    const OptimizeOptions& options = DefaultTrainingOptimizeOptions());

std::shared_ptr<hlir::framework::Graph> Optimize(
    frontend::Program* program,
    const std::unordered_set<std::string>& fetch_ids,
    common::Target target,
    const std::vector<std::string>& passes);

}  // namespace frontend
}  // namespace cinn
