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

class Conv2dBiasFusePass : public FusePassBase {
 protected:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  void TransFcBias(ir::Graph* graph, const std::string& mul_type) const;

  void FoldConv2dBias(ir::Graph* graph) const;
  /*
  For example:
                        x
                        |
                  elementwise_mul
                        |
                  elementwise_add
                        |
                       out
  ------------------------------------------------------
  After the pass is applied:
                        x
                        |
            bias --- scale_op --- scale
                        |
                       out
  */
  void FuseScaleOps(ir::Graph* graph) const;

  const std::string name_scope_{"conv2d_bias_fuse_pass"};
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
