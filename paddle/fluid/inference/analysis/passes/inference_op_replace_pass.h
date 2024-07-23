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

#pragma once

#include <string>
#include <vector>

#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/inference/analysis/analysis_pass.h"
#include "paddle/phi/common/place.h"

namespace paddle {
namespace inference {
namespace analysis {

/*
 * There are some ops (while, conditional_block_op etc) which have different
 * optimization points under prediction and training conditions.
 * So, We added the corresponding inference impl to these ops separately.
 * This pass replaces these ops with corresponding inference ops.
 */
struct Argument;

class InferenceOpReplacePass : public AnalysisPass {
 public:
  void RunImpl(Argument *argument) override;
  std::string repr() const override;
};

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
