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

#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/pass.h"

namespace paddle {
namespace framework {
namespace ir {

/*
 * Map matmul to mul, so the optimization can use fc_fuse_pass.
 * The mul op must satisfy the following conditions:
 * 1. the transpose_X and transpose_Y attrs are false
 * 2. the alpha attr is 1.0
 * 3. the rank of input X and Y is 2
 * 4. the next op of matmul is only elementwise_add
 *
 * Notice:
 *  the rank of input activation is obtained from var_desc,
 *  it maybe change in runtime.
 */
class Graph;

class MapMatmul2MulPass : public FusePassBase {
 public:
  virtual ~MapMatmul2MulPass() {}

 protected:
  void ApplyImpl(Graph* graph) const override;
};

/*
 * Fuse squeeze2+matmul to mul, so the optimization can use fc_fuse_pass.
 * The squeeze2 op must satisfy the following conditions:
 * 1. the rank of input X is 4
 * 2. the axis attr is [2, 3]
 * 3. the next op is only matmul
 *
 * The matmul op must satisfy the following conditions:
 * 1. the transpose_X and transpose_Y attrs are false
 * 2. the alpha attr is 1.0
 * 3. the rank of input X and Y is 2
 * 4. the next op of matmul is only elementwise_add
 *
 * Notice:
 *  the rank of input activation is obtained from var_desc,
 *  it maybe change in runtime. Therefore, the pass considers
 *  the above passes to reduce the impact on other models.
 */

class Squeeze2MatmulFusePass : public FusePassBase {
 public:
  virtual ~Squeeze2MatmulFusePass() {}

 protected:
  void ApplyImpl(Graph* graph) const override;
};

/*
 * Fuse reshape2+matmul to mul, so the optimization can use fc_fuse_pass.
 * The reshape2 op must satisfy the following conditions:
 * 1. reshape2 has one input node, which means it don't
 *    have Shape or ShapeTensor input
 * 2. the rank of input X is 4 and the last two dims of input X is 1
 * 3. the rank of shape attr is 2
 * 4. the next op is only matmul
 *
 * The matmul op must satisfy the following conditions:
 * 1. the transpose_X and transpose_Y attrs are false
 * 2. the alpha attr is 1.0
 * 3. the rank of input X and Y is 2
 * 4. the next op of matmul is only elementwise_add
 *
 * Notice:
 *  the shape and rank of input activation is obtained from var_desc,
 *  they maybe change in runtime. Therefore, the pass considers
 *  the above passes to reduce the impact on other models.
 */

class Reshape2MatmulFusePass : public FusePassBase {
 public:
  virtual ~Reshape2MatmulFusePass() {}

 protected:
  void ApplyImpl(Graph* graph) const override;
};

class Flatten2MatmulFusePass : public FusePassBase {
 public:
  virtual ~Flatten2MatmulFusePass() {}

 protected:
  void ApplyImpl(Graph* graph) const override;
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
