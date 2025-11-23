// Copyright (c) 2025 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/optim/ir_simplify.h"

#include <gtest/gtest.h>

#include "paddle/cinn/cinn.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"
#include "paddle/cinn/ir/utils/ir_nodes_collector.h"
#include "paddle/cinn/ir/utils/stmt_converter.h"
#include "paddle/cinn/utils/string.h"

namespace cinn {
namespace optim {

/*
serial for (i, 0ll, 32768ll) {
  serial for (j, 0ll, 16ll) {
    serial for (reduce_k_0, 0ll, 128ll) {
        var_18[i, j] = select((var_18[i, j] > var_17[i, j, reduce_k_0]),
var_18[i, j], var_17[i, j, reduce_k_0])
      }
    }
  }
}
*/
TEST(IRSimplifySelect, SimplifySelectToMax) {
  Context::Global().ResetNameId();

  // Create input IR matching the specified pattern
  const std::vector<ir::Expr> shape_2d = {ir::Expr(32768), ir::Expr(16)};
  const std::vector<ir::Expr> shape_3d = {
      ir::Expr(32768), ir::Expr(16), ir::Expr(128)};

  ir::Tensor var_17 =
      ir::_Tensor_::Make("var_17", ir::Float(32), shape_3d, shape_3d);
  var_17->WithBuffer("global", "var_17_buffer");

  ir::Tensor var_18 =
      ir::_Tensor_::Make("var_18", ir::Float(32), shape_2d, shape_2d);
  var_18->WithBuffer("global", "var_18_buffer");

  // Define loop variables
  ir::Var var_i = ir::Var(ir::Expr(0), ir::Expr(32768), "i");
  ir::Var var_j = ir::Var(ir::Expr(0), ir::Expr(16), "j");
  ir::Var var_reduce_k_0 = ir::Var(ir::Expr(0), ir::Expr(128), "reduce_k_0");

  // Create innermost reduction loop body
  ir::Expr reduce_body = ir::Store::Make(
      var_18,
      ir::Select::Make(
          ir::GT::Make(ir::Load::Make(var_18, {var_i, var_j}),
                       ir::Load::Make(var_17, {var_i, var_j, var_reduce_k_0})),
          ir::Load::Make(var_18, {var_i, var_j}),
          ir::Load::Make(var_17, {var_i, var_j, var_reduce_k_0})),
      {var_i, var_j});

  // Create reduction loop
  ir::Expr reduce_loop = ir::For::Make(var_reduce_k_0,
                                       ir::Expr(0),
                                       ir::Expr(128),
                                       ir::ForType::Serial,
                                       ir::DeviceAPI::Host,
                                       ir::Block::Make({reduce_body}));

  // Create j loop
  ir::Expr j_loop = ir::For::Make(var_j,
                                  ir::Expr(0),
                                  ir::Expr(16),
                                  ir::ForType::Serial,
                                  ir::DeviceAPI::Host,
                                  ir::Block::Make({reduce_loop}));

  // Create i loop
  ir::Expr i_loop = ir::For::Make(var_i,
                                  ir::Expr(0),
                                  ir::Expr(32768),
                                  ir::ForType::Serial,
                                  ir::DeviceAPI::Host,
                                  ir::Block::Make({j_loop}));

  // Final expression
  ir::Expr expr = ir::Block::Make({i_loop});

  VLOG(6) << "Before Simplify: " << expr;
  Simplify(&expr);
  VLOG(6) << "After Simplify: " << expr;

  // Expected output verification
  std::string expected_ir = R"ROC({
  serial for (i, 0, 32768)
  {
    serial for (j, 0, 16)
    {
      serial for (reduce_k_0, 0, 128)
      {
        var_18[i, j] = cinn_max(var_17[i, j, reduce_k_0], var_18[i, j])
      }
    }
  }
})ROC";

  EXPECT_EQ(utils::GetStreamCnt(expr), utils::Trim(expected_ir));
}

/*
serial for (i, 0ll, 32768ll) {
  serial for (j, 0ll, 16ll) {
    serial for (reduce_k_0, 0ll, 128ll) {
        var_18[i, j] = select((var_18[i, j] < var_17[i, j, reduce_k_0]),
var_18[i, j], var_17[i, j, reduce_k_0])
      }
    }
  }
}
*/
TEST(IRSimplifySelect, SimplifySelectToMin) {
  Context::Global().ResetNameId();

  // Create input IR matching the specified pattern
  const std::vector<ir::Expr> shape_2d = {ir::Expr(32768), ir::Expr(16)};
  const std::vector<ir::Expr> shape_3d = {
      ir::Expr(32768), ir::Expr(16), ir::Expr(128)};

  ir::Tensor var_17 =
      ir::_Tensor_::Make("var_17", ir::Float(32), shape_3d, shape_3d);
  var_17->WithBuffer("global", "var_17_buffer");

  ir::Tensor var_18 =
      ir::_Tensor_::Make("var_18", ir::Float(32), shape_2d, shape_2d);
  var_18->WithBuffer("global", "var_18_buffer");

  // Define loop variables
  ir::Var var_i = ir::Var(ir::Expr(0), ir::Expr(32768), "i");
  ir::Var var_j = ir::Var(ir::Expr(0), ir::Expr(16), "j");
  ir::Var var_reduce_k_0 = ir::Var(ir::Expr(0), ir::Expr(128), "reduce_k_0");

  // Create innermost reduction loop body
  ir::Expr reduce_body = ir::Store::Make(
      var_18,
      ir::Select::Make(
          ir::LT::Make(ir::Load::Make(var_18, {var_i, var_j}),
                       ir::Load::Make(var_17, {var_i, var_j, var_reduce_k_0})),
          ir::Load::Make(var_18, {var_i, var_j}),
          ir::Load::Make(var_17, {var_i, var_j, var_reduce_k_0})),
      {var_i, var_j});

  // Create reduction loop
  ir::Expr reduce_loop = ir::For::Make(var_reduce_k_0,
                                       ir::Expr(0),
                                       ir::Expr(128),
                                       ir::ForType::Serial,
                                       ir::DeviceAPI::Host,
                                       ir::Block::Make({reduce_body}));

  // Create j loop
  ir::Expr j_loop = ir::For::Make(var_j,
                                  ir::Expr(0),
                                  ir::Expr(16),
                                  ir::ForType::Serial,
                                  ir::DeviceAPI::Host,
                                  ir::Block::Make({reduce_loop}));

  // Create i loop
  ir::Expr i_loop = ir::For::Make(var_i,
                                  ir::Expr(0),
                                  ir::Expr(32768),
                                  ir::ForType::Serial,
                                  ir::DeviceAPI::Host,
                                  ir::Block::Make({j_loop}));

  // Final expression
  ir::Expr expr = ir::Block::Make({i_loop});

  VLOG(6) << "Before Simplify: " << expr;
  Simplify(&expr);
  VLOG(6) << "After Simplify: " << expr;

  // Expected output verification
  std::string expected_ir = R"ROC({
  serial for (i, 0, 32768)
  {
    serial for (j, 0, 16)
    {
      serial for (reduce_k_0, 0, 128)
      {
        var_18[i, j] = cinn_min(var_18[i, j], var_17[i, j, reduce_k_0])
      }
    }
  }
})ROC";

  EXPECT_EQ(utils::GetStreamCnt(expr), utils::Trim(expected_ir));
}

/*
serial for (i, 0ll, 32768ll)
{
    serial for (j, 0, 16)
    {
        serial for (j_0, 0, 128)
        {
            var_45[i, j, j_0)] = select(
                (var_18[i, ((((j * 128ll) + j_0) / 128ll) + 0ll)] <=
                 float32(3.4028234663852886e+38)),
                select(
                    (var_18[i, ((((j * 128ll) + j_0) / 128ll) + 0ll)] >=
                     float32(9.9999997473787516e-05)),
                    var_18[i, ((((j * 128ll) + j_0) / 128ll) + 0ll)],
                    float32(9.9999997473787516e-05)
                ),
                float32(3.4028234663852886e+38)
            )
        }
    }
}
*/
TEST(IRSimplifySelect, SimplifySelectToMinMax) {
  Context::Global().ResetNameId();

  // Create input IR matching the specified pattern
  const std::vector<ir::Expr> shape_2d = {ir::Expr(32768), ir::Expr(16)};
  const std::vector<ir::Expr> shape_3d = {
      ir::Expr(32768), ir::Expr(16), ir::Expr(128)};

  ir::Tensor var_18 =
      ir::_Tensor_::Make("var_18", ir::Float(32), shape_2d, shape_2d);
  var_18->WithBuffer("global", "var_18_buffer");

  ir::Tensor var_45 =
      ir::_Tensor_::Make("var_45", ir::Float(32), shape_3d, shape_3d);
  var_45->WithBuffer("global", "var_45_buffer");

  // Define loop variables
  ir::Var var_i = ir::Var(ir::Expr(0), ir::Expr(32768), "i");
  ir::Var var_j = ir::Var(ir::Expr(0), ir::Expr(16), "j");
  ir::Var var_j_0 = ir::Var(ir::Expr(0), ir::Expr(128), "j_0");

  // Create innermost loop body
  ir::Expr body = ir::Store::Make(
      var_45,
      ir::Select::Make(
          ir::LE::Make(
              ir::Load::Make(
                  var_18,
                  {var_i,
                   ir::Div::Make(
                       ir::Add::Make(ir::Mul::Make(var_j, ir::Expr(128)),
                                     var_j_0),
                       ir::Expr(128))}),
              ir::Expr(3.4028234663852886e+38f)),
          ir::Select::Make(
              ir::GE::Make(
                  ir::Load::Make(
                      var_18,
                      {var_i,
                       ir::Div::Make(
                           ir::Add::Make(ir::Mul::Make(var_j, ir::Expr(128)),
                                         var_j_0),
                           ir::Expr(128))}),
                  ir::Expr(9.9999997473787516e-05f)),
              ir::Load::Make(
                  var_18,
                  {var_i,
                   ir::Div::Make(
                       ir::Add::Make(ir::Mul::Make(var_j, ir::Expr(128)),
                                     var_j_0),
                       ir::Expr(128))}),
              ir::Expr(9.9999997473787516e-05f)),
          ir::Expr(3.4028234663852886e+38f)),
      {var_i, var_j, var_j_0});

  // Create j_0 loop
  ir::Expr j_0_loop = ir::For::Make(var_j_0,
                                    ir::Expr(0),
                                    ir::Expr(128),
                                    ir::ForType::Serial,
                                    ir::DeviceAPI::Host,
                                    ir::Block::Make({body}));

  // Create j loop
  ir::Expr j_loop = ir::For::Make(var_j,
                                  ir::Expr(0),
                                  ir::Expr(16),
                                  ir::ForType::Serial,
                                  ir::DeviceAPI::Host,
                                  ir::Block::Make({j_0_loop}));

  // Create i loop
  ir::Expr i_loop = ir::For::Make(var_i,
                                  ir::Expr(0),
                                  ir::Expr(32768),
                                  ir::ForType::Serial,
                                  ir::DeviceAPI::Host,
                                  ir::Block::Make({j_loop}));

  // Final expression
  ir::Expr expr = ir::Block::Make({i_loop});

  VLOG(6) << "Before Simplify: " << expr;
  Simplify(&expr);
  VLOG(6) << "After Simplify: " << expr;

  // Expected output verification
  std::string expected_ir = R"ROC({
  serial for (i, 0, 32768)
  {
    serial for (j, 0, 16)
    {
      serial for (j_0, 0, 128)
      {
        var_45[i, j, j_0] = cinn_min(cinn_max(var_18[i, (((j * 128) + j_0) / 128)], 9.99999975e-05f), 3.40282347e+38f)
      }
    }
  }
})ROC";

  EXPECT_EQ(utils::GetStreamCnt(expr), utils::Trim(expected_ir));
}
}  // namespace optim
}  // namespace cinn
