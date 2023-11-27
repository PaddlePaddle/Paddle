// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/pass.h"

namespace phi {
class DenseTensor;
}  // namespace phi

namespace paddle {
namespace framework {
class Scope;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace framework {
namespace ir {
/*
Squeeze and Excitaion Block Fusion for SE-ResNet
Origin subgraph
        Input
        |    \
        |     \
        |      \
        |       |
        |     Global Pooling
        |       |
        |       conv2d_xpu
        |       |
        |       |
        |       conv2d_xpu
        \       |
         \      |
           elementwise_mul
             |
           Output
------------------------------------------------------
After the pass is applied:

                          in_Input
            in_Filter      |     in_FilterMax
                      \    |    /
                        \  |   /
  in_Branch ------- squeeze_excitation_block ------ in_Bias
                           |
                           |
                           |
                      out_Output
*/
class SqueezeExcitationFusePass : public FusePassBase {
 protected:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  int ApplyImpl(ir::Graph* graph,
                const std::string& op_type,
                const std::string& act_type,
                bool with_branch,
                bool with_bias) const;

  const std::string name_scope_{"squeeze_excitation_fuse_pass"};
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
