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

#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/op_compat_sensible_pass.h"
#include "paddle/fluid/framework/scope.h"

namespace paddle {
namespace framework {
class Scope;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace framework {
namespace ir {

class Graph;
class Node;

static const char kParamScopeAttr[] = "__param_scope__";
static const char kFuseStatisAttr[] = "__fuse_statis__";
// When we use trt or other third_party lib, the parameters are managed by
// the lib, but not the fluid. So we need to record them to avoid duplicate
// allocation.
static const char kRepetitiveParamAttr[] = "__repetitive_param__";

// scale and zero point of the quantized/dequantized op should be removed in
// save_optimized_model_pass.
static const char kScaleAndZeroPointParamAttr[] =
    "__scale_and_zero_point_param__";

enum FuseOptions {
  DO_NOT_FUSE,  // fusing will not be done
  FUSE_NATIVE,  // fusing will be done without MKL-DNN
  FUSE_MKLDNN   // fusing will be done with MKL-DNN
};

class FusePassBase : public OpCompatSensiblePass {
 public:
  void Init(const std::string& repr, Graph* graph) const;
  Scope* param_scope() const;
  void AddStatis(int count_of_fused) const;

  virtual ~FusePassBase() {}

 protected:
  virtual FuseOptions FindFuseOption(const Node& node1,
                                     const Node& node2) const;

  mutable Graph* graph_;
  mutable std::string repr_;
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
