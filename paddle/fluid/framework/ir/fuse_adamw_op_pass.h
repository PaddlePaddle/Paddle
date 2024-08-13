// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
// Copyright (c) 2023 NVIDIA Authors. All Rights Reserved.
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

#include <mutex>
#include <string>
#include <unordered_set>

#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/pass.h"

namespace paddle {
namespace framework {
namespace ir {

class Graph;
class Node;

struct AdamWConfig {
  Node *first_lr = nullptr;
  Node *first_skip_update = nullptr;
  paddle::framework::BlockDesc *block = nullptr;
  int op_role = 0;
  float beta1 = 0.9;
  float beta2 = 0.99;
  float epsilon = 1e-8;
  float first_coeff = 0.0;
  bool use_global_beta_pow = false;
  bool replace_adamw = true;
  bool use_skip_update = false;
  bool with_decay = true;
  bool multi_precision = true;

  // Initialize the input and output names of adamw op and fused_adamw op
  const std::vector<std::string> inputs_name = {
      "Param", "Grad", "Moment1", "Moment2", "Beta1Pow", "Beta2Pow"};
  const std::vector<std::string> outputs_name = {
      "ParamOut", "Moment1Out", "Moment2Out", "Beta1PowOut", "Beta2PowOut"};
  const std::vector<std::string> replace_inputs_name = {
      "Params", "Grads", "Moments1", "Moments2", "Beta1Pows", "Beta2Pows"};
  const std::vector<std::string> replace_outputs_name = {"ParamsOut",
                                                         "Moments1Out",
                                                         "Moments2Out",
                                                         "Beta1PowsOut",
                                                         "Beta2PowsOut"};
};

class FuseAdamWPass : public FusePassBase {
 public:
  virtual ~FuseAdamWPass() {}

 protected:
  void ApplyImpl(ir::Graph *graph) const override;

  ir::Graph *FuseAdamWFun(ir::Graph *graph,
                          const bool with_decay,
                          const bool multi_precision) const;
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
