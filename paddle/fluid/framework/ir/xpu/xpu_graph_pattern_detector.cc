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

#include "paddle/fluid/framework/ir/xpu/xpu_graph_pattern_detector.h"

namespace paddle {
namespace framework {
namespace ir {
namespace patterns {
PDNode *patterns::DequantXPUAny::operator()() {
  auto *dequant_op =
      pattern->NewNode(dequant_op_repr())->assert_is_op("dequantize_xpu");

  auto *dequant_out = pattern->NewNode(dequant_out_repr())
                          ->AsOutput()
                          ->assert_is_op_output("dequantize_xpu", "y");

  auto *next_op = pattern->NewNode(next_op_repr())->assert_is_op();

  dequant_op->LinksTo({dequant_out});
  next_op->LinksFrom({dequant_out});

  return dequant_out;
}

PDNode *patterns::QuantXPUAny::operator()() {
  auto *quant_in = pattern->NewNode(quant_in_repr())
                       ->AsInput()
                       ->assert_is_op_input("quantize_xpu", "x");
  auto *quant_op =
      pattern->NewNode(quant_op_repr())->assert_is_op("quantize_xpu");

  auto *quant_out = pattern->NewNode(quant_out_repr())
                        ->AsOutput()
                        ->assert_is_op_output("quantize_xpu", "y");

  auto *next_op = pattern->NewNode(next_op_repr())->assert_is_op();

  quant_op->LinksFrom({quant_in}).LinksTo({quant_out});
  next_op->LinksFrom({quant_out});

  return quant_out;
}

PDNode *patterns::DequantQuantXPUAny::operator()() {
  auto *dequant_in = pattern->NewNode(dequant_in_repr())
                         ->AsInput()
                         ->assert_is_op_input("dequantize_xpu", "x");

  auto *dequant_op =
      pattern->NewNode(dequant_op_repr())->assert_is_op("dequantize_xpu");

  auto *dequant_out = pattern->NewNode(dequant_out_repr())
                          ->AsOutput()
                          ->assert_is_op_output("dequantize_xpu", "y");

  auto *quant_op = pattern->NewNode(quant_op_repr())
                       ->assert_is_op("quantize_xpu")
                       ->AsIntermediate();

  auto *quant_out = pattern->NewNode(quant_out_repr())
                        ->AsOutput()
                        ->assert_is_op_output("quantize_xpu");

  auto *next_op = pattern->NewNode(next_op_repr())->assert_is_op();

  dequant_op->LinksFrom({dequant_in}).LinksTo({dequant_out});
  quant_op->LinksFrom({dequant_out}).LinksTo({quant_out});
  next_op->LinksFrom({quant_out});

  return quant_out;
}

PDNode *patterns::OpDequantXPU::operator()() {
  auto any_op = pattern->NewNode(any_op_repr())->assert_is_op();
  auto *dequant_in = pattern->NewNode(dequant_in_repr())
                         ->assert_is_op_input("dequantize_xpu", "x");
  auto *dequant_op =
      pattern->NewNode(dequant_op_repr())->assert_is_op("dequantize_xpu");
  auto dequant_out = pattern->NewNode(dequant_out_repr())
                         ->AsOutput()
                         ->assert_is_op_output("dequantize_xpu", "y");

  any_op->LinksTo({dequant_in});
  dequant_op->LinksFrom({dequant_in}).LinksTo({dequant_out});
  return dequant_out;
}

PDNode *patterns::MultipleQuantizeXPU::operator()() {
  auto *prev_out = pattern->NewNode(prev_out_repr())->AsOutput();

  // find nodes that are inputs to quantize operators
  prev_out->assert_more([&](Node *node) {
    int counter = static_cast<int>(std::count_if(
        node->outputs.begin(), node->outputs.end(), [&](Node const *iter) {
          return iter && iter->IsOp() && iter->Op()->Type() == "quantize_xpu";
        }));
    return (counter > 1);
  });

  return prev_out;
}

PDNode *patterns::QuantConv2dFusionDequantXPU::operator()() {
  auto *quant_in = pattern->NewNode(quant_in_repr())
                       ->AsInput()
                       ->assert_is_op_input("quantize_xpu", "x");
  auto *quant_op =
      pattern->NewNode(quant_op_repr())->assert_is_op("quantize_xpu");

  auto *quant_out = pattern->NewNode(quant_out_repr())
                        ->AsOutput()
                        ->assert_is_op_output("quantize_xpu", "y")
                        ->assert_is_op_input("conv2d_xpu", "branch");

  auto *conv_op = pattern->NewNode(conv_op_repr())->assert_is_op("conv2d_xpu");
  quant_op->LinksFrom({quant_in}).LinksTo({quant_out});
  auto *conv_out = pattern->NewNode(conv_out_repr())
                       ->assert_is_op_output("conv2d_xpu", "out");
  conv_op->LinksFrom({quant_out}).LinksTo({conv_out});
  auto *dequant_op =
      pattern->NewNode(dequant_op_repr())->assert_is_op("dequantize_xpu");
  auto *dequant_out = pattern->NewNode(dequant_out_repr())
                          ->AsOutput()
                          ->assert_is_op_output("dequantize_xpu", "y");
  dequant_op->LinksFrom({conv_out}).LinksTo({dequant_out});
  return dequant_out;
}

}  // namespace patterns
}  // namespace ir
}  // namespace framework
}  // namespace paddle
