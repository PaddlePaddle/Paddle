//   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
#include <utility>
#include <vector>

#include "paddle/fluid/framework/details/build_strategy.h"
#include "paddle/fluid/framework/details/multi_devices_helper.h"
#include "paddle/fluid/framework/ir/graph.h"

namespace paddle {
namespace framework {
namespace details {

class FuseAdamOpPass : public ir::Pass {
 protected:
  std::unique_ptr<ir::Graph> ApplyImpl(
      std::unique_ptr<ir::Graph> graph) const override;

 private:
  void GetSpecifiedOpsAndVars(
      const std::string &op_type, const std::vector<std::string> &aux_vars_name,
      ir::Node *node, std::vector<ir::Node *> *ops,
      std::unordered_map<std::string, std::vector<std::string>> *aux_args_name)
      const;

  void FuseAdamOps(
      const std::unordered_map<std::string, std::vector<std::string>> &vars_set,
      const std::unordered_map<std::string, std::string> &fused_vars_name,
      const std::vector<ir::Node *> &adam_ops, ir::Graph *graph) const;

  void FuseScaleOps(const std::vector<std::string> &aux_var_set,
                    const std::string &fused_var_name,
                    const std::vector<ir::Node *> &adam_ops,
                    ir::Graph *graph) const;

  void SortVarsName(
      const std::string &str,
      std::unordered_map<std::string, std::vector<std::string>> *aux_var_set,
      std::vector<ir::Node *> *ops) const;

  void AppendAllocContinuousSpace(const std::vector<std::string> &args,
                                  const std::string &out_arg, bool copy_data,
                                  BlockDesc *global_block) const;
};

}  // namespace details
}  // namespace framework
}  // namespace paddle
