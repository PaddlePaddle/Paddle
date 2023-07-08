// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#include "test/cpp/cinn/program_builder.h"

namespace cinn {
namespace tests {

/*
 * Add --* Multiply --* Add --* Relu
 */
class BiasBnReLUBuilder : public ProgramBuilder {
 public:
  BiasBnReLUBuilder() : ProgramBuilder("bias_bn_relu_builder") {}
  frontend::Program Build(const std::vector<VariableInfo>& inputs_varinfo,
                          const utils::AttributeMap& attrs = {}) {
    CHECK_EQ(inputs_varinfo.size(), 4);
    auto conv_output = builder_.CreateInput(
        inputs_varinfo[0].type, inputs_varinfo[0].shape, inputs_varinfo[0].id);
    auto bias = builder_.CreateInput(
        inputs_varinfo[1].type, inputs_varinfo[1].shape, inputs_varinfo[1].id);
    auto bn_scale = builder_.CreateInput(
        inputs_varinfo[2].type, inputs_varinfo[2].shape, inputs_varinfo[2].id);
    auto bn_offset = builder_.CreateInput(
        inputs_varinfo[3].type, inputs_varinfo[3].shape, inputs_varinfo[3].id);

    auto bias_add = builder_.Add(conv_output, bias);
    auto bn_mul = builder_.Multiply(bias_add, bn_scale);
    auto bn_add = builder_.Add(bn_mul, bn_offset);
    builder_.Relu(bn_add);
    return builder_.Build();
  }
};

/*
 * Exp --* Add
 *    \
 *     --* Multiply
 */
class ExpTwoConsumersOpBuilder : public ProgramBuilder {
 public:
  ExpTwoConsumersOpBuilder() : ProgramBuilder("exp_two_consumers_builder") {}
  frontend::Program Build(const std::vector<VariableInfo>& inputs_varinfo,
                          const utils::AttributeMap& attrs = {}) {
    CHECK_EQ(inputs_varinfo.size(), 1);
    auto x = builder_.CreateInput(
        inputs_varinfo[0].type, inputs_varinfo[0].shape, inputs_varinfo[0].id);
    auto exp_x = builder_.Exp(x);
    auto add_x = builder_.Add(exp_x, x);
    auto mul_1 = builder_.Multiply(exp_x, add_x);
    return builder_.Build();
  }
};

/*
 * Gather --* Add --* Subtract
 *                    *
 *                   /
 *            Gather
 */
class GatherAddSubBuilder : public ProgramBuilder {
 public:
  GatherAddSubBuilder() : ProgramBuilder("gather_add_sub_builder") {}
  frontend::Program Build(const std::vector<VariableInfo>& inputs_varinfo,
                          const utils::AttributeMap& attrs = {}) {
    CHECK_EQ(inputs_varinfo.size(), 2);
    auto x = builder_.CreateInput(
        inputs_varinfo[0].type, inputs_varinfo[0].shape, inputs_varinfo[0].id);
    auto y = builder_.CreateInput(
        inputs_varinfo[1].type, inputs_varinfo[1].shape, inputs_varinfo[1].id);
    auto input_x_shape = inputs_varinfo[0].shape;
    auto where_x_0 = builder_.Gather(
        x, builder_.FillConstant({input_x_shape[0]}, 0, "constant_idx_first"));
    auto where_x_last = builder_.Gather(
        x,
        builder_.FillConstant(
            {input_x_shape[0]}, input_x_shape[0] - 1, "constant_idx_last"));
    auto add_1 = builder_.Add(where_x_0, y);
    builder_.Subtract(where_x_last, add_1);
    return builder_.Build();
  }
};

/*
 * FillConstant --* Add
 */
class FillConstantAddBuilder : public ProgramBuilder {
 public:
  FillConstantAddBuilder() : ProgramBuilder("fill_constant_add_builder") {}
  frontend::Program Build(const std::vector<VariableInfo>& inputs_varinfo,
                          const utils::AttributeMap& attrs = {}) {
    CHECK_EQ(inputs_varinfo.size(), 1);
    auto x = builder_.CreateInput(
        inputs_varinfo[0].type, inputs_varinfo[0].shape, inputs_varinfo[0].id);
    auto fill_constant =
        builder_.FillConstant(inputs_varinfo[0].shape, 1.0f, "fill_constant");
    builder_.Add(x, fill_constant);
    return builder_.Build();
  }
};

}  // namespace tests
}  // namespace cinn
