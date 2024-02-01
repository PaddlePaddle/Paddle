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

#include "paddle/fluid/framework/ir/graph_pattern_detector.h"

namespace paddle {
namespace framework {
namespace ir {
namespace patterns {

// Dequantize + anyOP
// This quantize is used for getting number of ops the Dequantize's
// output is an input to.
struct DequantXPUAny : public PatternBase {
  DequantXPUAny(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "dequant_xpu_any") {}
  PDNode* operator()();

  PATTERN_DECL_NODE(dequant_op);
  PATTERN_DECL_NODE(dequant_out);
  PATTERN_DECL_NODE(next_op);
};

// Quantize + anyOP
struct QuantXPUAny : public PatternBase {
  QuantXPUAny(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "quant_xpu_any") {}
  PDNode* operator()();

  PATTERN_DECL_NODE(quant_in);
  PATTERN_DECL_NODE(quant_op);
  PATTERN_DECL_NODE(quant_out);
  PATTERN_DECL_NODE(next_op);
};

// Dequantize + Quantize + anyOP
// This pattern is used for squashing the dequantize-quantize pairs.
struct DequantQuantXPUAny : public PatternBase {
  DequantQuantXPUAny(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "dequant_quant_xpu_any") {}
  PDNode* operator()();

  PATTERN_DECL_NODE(dequant_in);
  PATTERN_DECL_NODE(dequant_max_in);
  PATTERN_DECL_NODE(dequant_op);
  PATTERN_DECL_NODE(dequant_out);
  PATTERN_DECL_NODE(quant_max_in);
  PATTERN_DECL_NODE(quant_op);
  PATTERN_DECL_NODE(quant_out);
  PATTERN_DECL_NODE(next_op);
};

// Op + Dequant
// named nodes:
// any_op, dequant_in
// dequant_op, dequant_out
struct OpDequantXPU : public PatternBase {
  OpDequantXPU(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "op_dequant_xpu") {}

  PDNode* operator()();

  PATTERN_DECL_NODE(any_op);
  PATTERN_DECL_NODE(dequant_in);
  PATTERN_DECL_NODE(dequant_max_in);
  PATTERN_DECL_NODE(dequant_op);
  PATTERN_DECL_NODE(dequant_out);
};

// anyOp + more then one quantize op
// This pattern is used for squashing multiple quantize with the same scale.
struct MultipleQuantizeXPU : public PatternBase {
  MultipleQuantizeXPU(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "multiple_quantize_xpu") {}
  PDNode* operator()();

  PATTERN_DECL_NODE(prev_out);
};

// quantize_xpu(branch_input) + conv2d_xpu + dequantize_xpu
struct QuantConv2dFusionDequantXPU : public PatternBase {
  QuantConv2dFusionDequantXPU(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "quant_conv2d_fusion_dequant_xpu") {}
  PDNode* operator()();

  PATTERN_DECL_NODE(quant_in);
  PATTERN_DECL_NODE(quant_op);
  PATTERN_DECL_NODE(quant_out);
  PATTERN_DECL_NODE(conv_op);
  PATTERN_DECL_NODE(conv_out);
  PATTERN_DECL_NODE(dequant_op);
  PATTERN_DECL_NODE(dequant_out);
};

}  // namespace patterns
}  // namespace ir
}  // namespace framework
}  // namespace paddle
