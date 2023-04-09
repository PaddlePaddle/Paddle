// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
// Copyright (c) 2022 NVIDIA Authors. All Rights Reserved.
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

/*
 * Fuse Linear in ColumnParallel Linear and RowParallel Linear
 */
class Graph;
class Node;

class FuseLinearWithMPScalePass : public FusePassBase {
 public:
  virtual ~FuseLinearWithMPScalePass() {}

 protected:
  void ApplyImpl(ir::Graph *graph) const override;

  ir::Graph *FusedLinearFwd(ir::Graph *graph, bool is_training) const;
  ir::Graph *FusedLinearBwd(ir::Graph *graph, bool without_x_gradient) const;

  ir::Graph *FusedLinearWithMpScaleFwd(ir::Graph *graph) const;
  ir::Graph *FusedLinearWithMpScaleBwd(ir::Graph *graph) const;

 private:
  bool IsGemmFromLinear_(const std::vector<int64_t> &x_shape,
                         const std::vector<int64_t> &w_shape,
                         OpDesc *matmul_v2_op) const;
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
