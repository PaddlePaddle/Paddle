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
#include "paddle/fluid/framework/ir/graph.h"

namespace paddle {
namespace framework {
namespace ir {

/*
 * \brief   Fuse conv1x1 and depthwise conv operators
 */
class Conv1x1DepthwiseConvOneDNNFusePass : public FusePassBase {
 public:
  virtual ~Conv1x1DepthwiseConvOneDNNFusePass() {}
  Conv1x1DepthwiseConvOneDNNFusePass();

 protected:
  void ApplyImpl(ir::Graph* graph) const override;
  void FuseConvDepthWise(const std::string& conv_type,
                         bool with_bias,
                         Graph* graph) const;
  const std::string name_scope_{"conv1x1_depthwise_conv"};
  std::vector<std::string> conv_types = {"conv_2d", "fused_conv2d"};
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
