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

class GatherSqueezePass : public FusePassBase {
 protected:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  /*
  add transpose2 before gather + squeeze2
  For example:
  graph:
                        x
                        |
                      gather (axis = -1)
                        |
                    squeeze2 (axis = -1)
                        |
                      output
  ------------------------------------------------------
  After the pass is applied:
                        x
                        |
                    transpose2 (2, 0, 1)
                        |
                      gather (axis = 0)
                        |
                    squeeze2 (axis = 0)
                        |
                      output
  */
  void AddTranspose(ir::Graph* graph) const;

  const std::string name_scope_{"gather_squeeze_pass"};
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
