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

#include "paddle/cinn/optim/eliminate_common_factor_of_local_index.h"

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
{
  serial for (i_0, 0, 32) {
    serial for (j_0, 0, 4) {
      var_local[((i_0 * 3) + j_0), ((j_0 * 32) / 128)] =
      var_global_in[i_0, ((j_0 * 32) / 128)]
      var_global_out[((i_0 * 3) + j_0), j_0] =
      var_local[((i_0 * 3) + j_0), ((j_0 * 32) / 128)]
    }
  }
}
*/
TEST(EliminateCommonFactorOfLocalIndex, SimplifyLocalIndex) {
  Context::Global().ResetNameId();

  // Create input IR matching the specified pattern
  const std::vector<ir::Expr> shape = {ir::Expr(128), ir::Expr(128)};
  //   const std::vector<ir::Expr> indices = {ir::Expr(0)};
  ir::Tensor var_global_in =
      ir::_Tensor_::Make("var_global_in", ir::Float(32), shape, shape);
  var_global_in->WithBuffer("global", "var_global_in_buffer");
  ir::Tensor var_local_tensor =
      ir::_Tensor_::Make("var_local", ir::Float(32), shape, shape);
  var_local_tensor->WithBuffer("local", "var_local_buffer");
  ir::Tensor var_global_out =
      ir::_Tensor_::Make("var_global_out", ir::Float(32), shape, shape);
  var_global_out->WithBuffer("global", "var_global_out_buffer");

  ir::Var var_i_0 = ir::Var(ir::Expr(0), ir::Expr(32), "i_0");
  ir::Var var_j_0 = ir::Var(ir::Expr(0), ir::Expr(4), "j_0");

  std::vector<ir::Expr> block_contents = {
      ir::Store::Make(
          var_local_tensor,
          ir::Load::Make(var_global_in, {var_i_0, (var_j_0 * 32) / 128}),
          {var_i_0 * 3 + var_j_0, (var_j_0 * 32) / 128}),
      ir::Store::Make(
          var_global_out,
          ir::Load::Make(var_local_tensor,
                         {var_i_0 * 3 + var_j_0, (var_j_0 * 32) / 128}),
          {var_i_0 * 3 + var_j_0, var_j_0})};

  ir::Expr inner_loop_body = ir::Block::Make(block_contents);

  std::vector<ir::Expr> j_loop_contents = {inner_loop_body};
  ir::Expr j_loop = ir::For::Make(var_j_0,
                                  ir::Expr(0),
                                  ir::Expr(4),
                                  ir::ForType::Serial,
                                  ir::DeviceAPI::Host,
                                  ir::Block::Make(j_loop_contents));

  std::vector<ir::Expr> i_loop_contents = {j_loop};
  ir::Expr loop_body = ir::For::Make(var_i_0,
                                     ir::Expr(0),
                                     ir::Expr(32),
                                     ir::ForType::Serial,
                                     ir::DeviceAPI::Host,
                                     ir::Block::Make(i_loop_contents));

  std::vector<ir::Expr> ij_loop_contents = {loop_body};
  ir::Expr expr = ir::Block::Make(ij_loop_contents);

  ir::stmt::BlockRef block = ir::ConvertExprBlockToStmtBlock(expr);
  VLOG(6) << "Before EliminateCommonFactorOfLocalIndex: " << block;
  EliminateCommonFactorOfLocalIndex(block);
  VLOG(6) << "After EliminateCommonFactorOfLocalIndex: " << block;

  // Expected output verification
  std::string expected_ir = R"ROC({
  serial for (i_0, 0, 32) {
    serial for (j_0, 0, 4) {
      var_local[i_0, j_0] = var_global_in[i_0, ((j_0 * 32) / 128)]
      var_global_out[((i_0 * 3) + j_0), j_0] = var_local[i_0, j_0]
    }
  }
})ROC";

  EXPECT_EQ(utils::GetStreamCnt(block), utils::Trim(expected_ir));
}

/*
{
  serial for (i_0, 0, 32) {
    serial for (j_0, 0, 4) {
      var_local[(i_0 * 3), ((j_0 * 32) / 128)] =
      var_global_in[i_0, ((j_0 * 32) / 128)]
      var_global_out[((i_0 * 3) + j_0), j_0] =
      var_local[(i_0 * 3), ((j_0 * 32) / 128)]
    }
  }
}
*/
TEST(EliminateCommonFactorOfLocalIndex, SimplifyLocalIndexWithZeroIndex) {
  Context::Global().ResetNameId();

  // Create input IR matching the specified pattern
  const std::vector<ir::Expr> shape = {ir::Expr(128), ir::Expr(128)};
  //   const std::vector<ir::Expr> indices = {ir::Expr(0)};
  ir::Tensor var_global_in =
      ir::_Tensor_::Make("var_global_in", ir::Float(32), shape, shape);
  var_global_in->WithBuffer("global", "var_global_in_buffer");
  ir::Tensor var_local_tensor =
      ir::_Tensor_::Make("var_local", ir::Float(32), shape, shape);
  var_local_tensor->WithBuffer("local", "var_local_buffer");
  ir::Tensor var_global_out =
      ir::_Tensor_::Make("var_global_out", ir::Float(32), shape, shape);
  var_global_out->WithBuffer("global", "var_global_out_buffer");

  ir::Var var_i_0 = ir::Var(ir::Expr(0), ir::Expr(32), "i_0");
  ir::Var var_j_0 = ir::Var(ir::Expr(0), ir::Expr(4), "j_0");

  std::vector<ir::Expr> block_contents = {
      ir::Store::Make(
          var_local_tensor,
          ir::Load::Make(var_global_in, {var_i_0, var_j_0 * 32 / 128}),
          {var_i_0 * 3, var_j_0 * 32 / 128}),
      ir::Store::Make(
          var_global_out,
          ir::Load::Make(var_local_tensor, {var_i_0 * 3, var_j_0 * 32 / 128}),
          {var_i_0 * 3 + var_j_0, var_j_0})};

  ir::Expr inner_loop_body = ir::Block::Make(block_contents);

  std::vector<ir::Expr> j_loop_contents = {inner_loop_body};
  ir::Expr j_loop = ir::For::Make(var_j_0,
                                  ir::Expr(0),
                                  ir::Expr(4),
                                  ir::ForType::Serial,
                                  ir::DeviceAPI::Host,
                                  ir::Block::Make(j_loop_contents));

  std::vector<ir::Expr> i_loop_contents = {j_loop};
  ir::Expr loop_body = ir::For::Make(var_i_0,
                                     ir::Expr(0),
                                     ir::Expr(32),
                                     ir::ForType::Serial,
                                     ir::DeviceAPI::Host,
                                     ir::Block::Make(i_loop_contents));

  std::vector<ir::Expr> ij_loop_contents = {loop_body};
  ir::Expr expr = ir::Block::Make(ij_loop_contents);

  ir::stmt::BlockRef block = ir::ConvertExprBlockToStmtBlock(expr);
  VLOG(6) << "Before EliminateCommonFactorOfLocalIndex: " << block;
  EliminateCommonFactorOfLocalIndex(block);
  VLOG(6) << "After EliminateCommonFactorOfLocalIndex: " << block;

  // Expected output verification
  std::string expected_ir = R"ROC({
  serial for (i_0, 0, 32) {
    serial for (j_0, 0, 4) {
      var_local[0, i_0] = var_global_in[i_0, ((j_0 * 32) / 128)]
      var_global_out[((i_0 * 3) + j_0), j_0] = var_local[0, i_0]
    }
  }
})ROC";

  EXPECT_EQ(utils::GetStreamCnt(block), utils::Trim(expected_ir));
}
}  // namespace optim
}  // namespace cinn
