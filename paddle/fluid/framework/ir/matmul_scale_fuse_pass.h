// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

class Graph;

/*
 * Fuse the matmul and scale to a matmul.
 */
class MatmulScaleFusePass : public FusePassBase {
 public:
  MatmulScaleFusePass();
  virtual ~MatmulScaleFusePass() {}

 protected:
  void ApplyImpl(Graph* graph) const override;
};

/*
 * Fuse the matmul_v2 and scale to a matmul_v2.
 */
class MatmulV2ScaleFusePass : public FusePassBase {
 public:
  MatmulV2ScaleFusePass();
  virtual ~MatmulV2ScaleFusePass() {}

 protected:
  void ApplyImpl(Graph* graph) const override;
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
