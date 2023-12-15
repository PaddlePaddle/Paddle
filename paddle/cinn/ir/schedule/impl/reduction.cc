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

#include "paddle/cinn/common/macros.h"
#include "paddle/cinn/ir/schedule/factorize_reduction.h"
#include "paddle/cinn/ir/schedule/impl/ir_schedule.h"

/** \brief A macro that guards the beginning of each implementation of schedule
 */
#define CINN_IR_SCHEDULE_BEGIN() try {
/**
 * \brief A macro that pairs with `CINN_IR_SCHEDULE_BEGIN`, handling potential
 * errors and error message printing.
 * @param primitive A string representing the kind of schedule primitive.
 * @param err_msg_level A ScheduleErrorMessageLevel enum, level of error message
 * printing
 */
#define CINN_IR_SCHEDULE_END(err_msg_level)                    \
  }                                                            \
  catch (const utils::ErrorHandler& err_hanlder) {             \
    CINN_THROW(err_hanlder.FormatErrorMessage(err_msg_level)); \
  }

namespace cinn {
namespace ir {

Expr DyScheduleImpl::Rfactor(const Expr& rf_loop, int rf_axis) {
  CHECKRfactorValidation(rf_loop, rf_axis);
  // get root ScheduleBlockRealize
  Expr root = GetRootBlock(rf_loop);
  // create all stmts after rfactor transformation
  RfCreater rf_create(root, rf_loop, rf_axis);
  // return new created rfactor tensor
  return rf_create.CreateRfAllStmts();
}

Expr DyScheduleImpl::FactorizeReduction(const Expr& rf_loop, int rf_axis) {
  std::string primitive = "FactorizeReduction";
  // Get child block of the rf_loop and check.
  std::vector<Expr> blocks = GetChildBlocks(rf_loop);
  if (blocks.size() != 1) {
    std::ostringstream os;
    os << "The rf_loop is required to have only one child block, but got "
       << blocks.size() << std::endl;
    throw IRScheduleErrorHandler(primitive, os.str(), this->module_expr_);
  }
  Expr original_block = blocks.at(0);
  Expr root_block = GetRootBlock(original_block);
  // TODO(BiynXu): Add CheckReductionBlock()

  // Collect the loops of the block.
  // Construct a map from loop var names to corresponding loops.
  std::vector<Expr> original_loops = this->GetLoops(original_block);
  CHECK_GT(original_loops.size(), 0);
  VLOG(3) << "before FactorizeReduction, original computational body of the "
             "reduction is:\n"
          << original_loops[0];
  std::map<Var, Expr, CompVar> var2loops;
  for (const Expr& loop : original_loops) {
    var2loops[loop.As<For>()->loop_var] = loop;
  }

  // Get original stmt of reduction update and original store tensor.
  Expr original_update_body = original_block.As<ir::ScheduleBlockRealize>()
                                  ->schedule_block.As<ir::ScheduleBlock>()
                                  ->body;
  Expr original_update_stmt;
  CHECK(original_update_body.As<Block>() || original_update_body.As<Store>());
  if (original_update_body.As<Block>()) {
    CHECK_EQ(original_update_body.As<Block>()->stmts.size(), 1);
    original_update_stmt = original_update_body.As<Block>()->stmts[0];
  } else if (original_update_body.As<Store>()) {
    original_update_stmt = original_update_body;
  }
  Tensor original_tensor =
      original_update_stmt.As<Store>()->tensor.as_tensor_ref();

  // Create new blocks and loops.
  Tensor rf_tensor = CreateRFTensor(original_tensor, rf_loop, rf_axis);
  RFBlockCreater rf_block_creater(original_block,
                                  original_loops,
                                  rf_loop,
                                  original_update_stmt,
                                  rf_tensor,
                                  var2loops,
                                  rf_axis);
  rf_block_creater.CreateBlock();
  RBBlockCreater wb_block_creater(original_block,
                                  original_loops,
                                  rf_loop,
                                  original_update_stmt,
                                  rf_tensor,
                                  rf_block_creater.rf_tensor_access_indices_,
                                  rf_block_creater.rf_var_);
  wb_block_creater.CreateBlock();

  Expr rf_body = rf_block_creater.CreateLoops();
  Expr wb_body = wb_block_creater.CreateLoops();

  Expr new_computational_body = Block::Make({rf_body, wb_body});

  // Replace and update the AST.
  this->Replace(original_loops[0], new_computational_body);
  VLOG(3) << "After FactorizeReduction, new computational body of the "
             "reduction is:\n"
          << new_computational_body;
  return rf_tensor;
}

}  // namespace ir
}  // namespace cinn

namespace cinn {
namespace ir {

Expr StScheduleImpl::Rfactor(const Expr& rf_loop, int rf_axis) {
  CHECKRfactorValidation(rf_loop, rf_axis);
  // get root ScheduleBlockRealize
  Expr root = GetRootBlock(rf_loop);
  // create all stmts after rfactor transformation
  RfCreater rf_create(root, rf_loop, rf_axis);
  // return new created rfactor tensor
  return rf_create.CreateRfAllStmts();
}

Expr StScheduleImpl::FactorizeReduction(const Expr& rf_loop, int rf_axis) {
  std::string primitive = "FactorizeReduction";
  // Get child block of the rf_loop and check.
  std::vector<Expr> blocks = GetChildBlocks(rf_loop);
  if (blocks.size() != 1) {
    std::ostringstream os;
    os << "The rf_loop is required to have only one child block, but got "
       << blocks.size() << std::endl;
    throw IRScheduleErrorHandler(primitive, os.str(), this->module_expr_);
  }
  Expr original_block = blocks.at(0);
  Expr root_block = GetRootBlock(original_block);
  // TODO(BiynXu): Add CheckReductionBlock()

  // Collect the loops of the block.
  // Construct a map from loop var names to corresponding loops.
  std::vector<Expr> original_loops = this->GetLoops(original_block);
  CHECK_GT(original_loops.size(), 0);
  VLOG(3) << "before FactorizeReduction, original computational body of the "
             "reduction is:\n"
          << original_loops[0];
  std::map<Var, Expr, CompVar> var2loops;
  for (const Expr& loop : original_loops) {
    var2loops[loop.As<For>()->loop_var] = loop;
  }

  // Get original stmt of reduction update and original store tensor.
  Expr original_update_body = original_block.As<ir::ScheduleBlockRealize>()
                                  ->schedule_block.As<ir::ScheduleBlock>()
                                  ->body;
  Expr original_update_stmt;
  CHECK(original_update_body.As<Block>() || original_update_body.As<Store>());
  if (original_update_body.As<Block>()) {
    CHECK_EQ(original_update_body.As<Block>()->stmts.size(), 1);
    original_update_stmt = original_update_body.As<Block>()->stmts[0];
  } else if (original_update_body.As<Store>()) {
    original_update_stmt = original_update_body;
  }
  Tensor original_tensor =
      original_update_stmt.As<Store>()->tensor.as_tensor_ref();

  // Create new blocks and loops.
  Tensor rf_tensor = CreateRFTensor(original_tensor, rf_loop, rf_axis);
  RFBlockCreater rf_block_creater(original_block,
                                  original_loops,
                                  rf_loop,
                                  original_update_stmt,
                                  rf_tensor,
                                  var2loops,
                                  rf_axis);
  rf_block_creater.CreateBlock();
  RBBlockCreater wb_block_creater(original_block,
                                  original_loops,
                                  rf_loop,
                                  original_update_stmt,
                                  rf_tensor,
                                  rf_block_creater.rf_tensor_access_indices_,
                                  rf_block_creater.rf_var_);
  wb_block_creater.CreateBlock();

  Expr rf_body = rf_block_creater.CreateLoops();
  Expr wb_body = wb_block_creater.CreateLoops();

  Expr new_computational_body = Block::Make({rf_body, wb_body});

  // Replace and update the AST.
  this->Replace(original_loops[0], new_computational_body);
  VLOG(3) << "After FactorizeReduction, new computational body of the "
             "reduction is:\n"
          << new_computational_body;
  return rf_tensor;
}

}  // namespace ir
}  // namespace cinn
