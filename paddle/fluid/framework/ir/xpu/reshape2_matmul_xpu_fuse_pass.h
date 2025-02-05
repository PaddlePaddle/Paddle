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

class Squeeze2MatmulXPUFusePass : public FusePassBase {
 protected:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  void FuseSqueeze2Matmul(ir::Graph* graph) const;
  const std::string name_scope_{"squeeze2_matmul_xpu_fuse_pass"};
};

class Reshape2MatmulXPUFusePass : public FusePassBase {
 protected:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  void FuseReshape2Matmul(ir::Graph* graph) const;
  const std::string name_scope_{"reshape2_matmul_xpu_fuse_pass"};
};

class MapMatmulV2ToMatmulXPUPass : public FusePassBase {
 protected:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  void MapMatmulV2ToMatmul(ir::Graph* graph) const;
  const std::string name_scope_{"map_matmulv2_to_matmul_xpu_pass"};
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
