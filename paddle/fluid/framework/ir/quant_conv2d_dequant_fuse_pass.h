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
#include <memory>

#include "paddle/fluid/framework/ir/fuse_pass_base.h"

namespace paddle {
namespace framework {
namespace ir {

///
/// Fuse quant + conv2d/depthwise_conv2d/matrix_multiply + dequant
///
class QuantDequantFusePass : public FusePassBase {
 public:
  QuantDequantFusePass();
  virtual ~QuantDequantFusePass() {}

 protected:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  void DeleteQuant(ir::Graph* graph,
                   Scope* scope,
                   const std::string& quant_type) const;
  void FuseDequant(ir::Graph* graph,
                   Scope* scope,
                   const std::string& quantized_op_type,
                   const std::string& dequant_type) const;
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
