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
#include "paddle/cinn/ir/schedule/impl/ir_schedule.h"
#include "paddle/cinn/runtime/intrinsic.h"
#include "paddle/common/enforce.h"
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
#define CINN_IR_SCHEDULE_END(err_msg_level)              \
  }                                                      \
  catch (const utils::ErrorHandler& err_handler) {       \
    PADDLE_THROW(::common::errors::Fatal(                \
        err_handler.FormatErrorMessage(err_msg_level))); \
  }

namespace cinn {
namespace ir {

Expr DyScheduleImpl::CacheRead(const Expr& block,
                               int read_buffer_index,
                               const std::string& memory_type) {
  CINN_IR_SCHEDULE_BEGIN();
  std::string primitive = "CacheRead";

  PADDLE_ENFORCE_NOT_NULL(
      block.As<ScheduleBlockRealize>(),
      ::common::errors::InvalidArgument([&]() {
        std::ostringstream os;
        os << "[IRScheduleError] An error occurred in the schedule primitive <"
           << primitive << ">.\n"
           << "[Error info] Expr param(block) is not a ScheduleBlockRealize!\n"
           << "[Expr info] The Expr of current schedule is "
           << module_expr_.GetExprs() << ".";
        return os.str();
      }()));

  auto root = GetRootBlock(block);
  ChangeBodyToBlock::Change(&root);
  Expr read_expr = GetNthAccessExpr(block, read_buffer_index, false);

  PADDLE_ENFORCE_NOT_NULL(
      block.As<ScheduleBlockRealize>(),
      ::common::errors::InvalidArgument([&]() {
        std::ostringstream os;
        os << "[IRScheduleError] An error occurred in the schedule primitive <"
           << primitive << ">.\n"
           << "[Error info] The read_expr is not a Load!\n"
           << "[Expr info] The Expr of current schedule is "
           << module_expr_.GetExprs() << ".";
        return os.str();
      }()));

  auto tensor_indices = read_expr.As<ir::Load>()->indices;
  CacheBlockInfo info;
  info.read_tensor = read_expr.As<ir::Load>()->tensor.as_tensor_ref();
  info.write_tensor = MakeCacheTensor(info.read_tensor, memory_type);
  info.alloc = info.write_tensor;

  auto read_ranges =
      CalculateTensorRegions(block, tensor_indices, info.read_tensor, root);
  auto new_block =
      MakeCacheBlock(read_ranges, &info, memory_type, this->GetDeviceAPI());
  FindInsertionPoint(root, &info, false);
  auto new_root = CacheReadRewriter::Rewrite(root, &info);
  this->Replace(
      root.As<ScheduleBlockRealize>()->schedule_block.As<ScheduleBlock>()->body,
      new_root.As<ScheduleBlockRealize>()
          ->schedule_block.As<ScheduleBlock>()
          ->body);
  return new_block;
  CINN_IR_SCHEDULE_END(this->err_msg_level_);
}

Expr DyScheduleImpl::CacheWrite(const Expr& block,
                                int write_buffer_index,
                                const std::string& memory_type) {
  CINN_IR_SCHEDULE_BEGIN();
  std::string primitive = "CacheWrite";

  PADDLE_ENFORCE_NOT_NULL(
      block.As<ScheduleBlockRealize>(),
      ::common::errors::InvalidArgument([&]() {
        std::ostringstream os;
        os << "[IRScheduleError] An error occurred in the schedule primitive <"
           << primitive << ">.\n"
           << "[Error info] Expr param(block) is not a ScheduleBlockRealize!\n"
           << "[Expr info] The Expr of current schedule is "
           << module_expr_.GetExprs() << ".";
        return os.str();
      }()));

  auto root = GetRootBlock(block);
  ChangeBodyToBlock::Change(&root);
  Expr write_expr = GetNthAccessExpr(block, write_buffer_index, true);

  PADDLE_ENFORCE_NOT_NULL(
      write_expr.As<ir::Store>(), ::common::errors::InvalidArgument([&]() {
        std::ostringstream os;
        os << "[IRScheduleError] An error occurred in the schedule primitive <"
           << primitive << ">.\n"
           << "[Error info] The write_expr is not a Store!\n"
           << "[Expr info] The Expr of current schedule is "
           << module_expr_.GetExprs() << ".";
        return os.str();
      }()));

  Tensor write_tensor = write_expr.As<ir::Store>()->tensor.as_tensor_ref();
  auto tensor_indices = write_expr.As<ir::Store>()->indices;
  CacheBlockInfo info;
  info.read_tensor = MakeCacheTensor(write_tensor, memory_type);
  info.write_tensor = write_tensor;
  info.alloc = info.read_tensor;
  auto write_ranges =
      CalculateTensorRegions(block, tensor_indices, info.write_tensor, root);
  auto new_block =
      MakeCacheBlock(write_ranges, &info, memory_type, this->GetDeviceAPI());
  FindInsertionPoint(root, &info, true);

  auto new_root = CacheWriteRewriter::Rewrite(root, &info);
  this->Replace(
      root.As<ScheduleBlockRealize>()->schedule_block.As<ScheduleBlock>()->body,
      new_root.As<ScheduleBlockRealize>()
          ->schedule_block.As<ScheduleBlock>()
          ->body);

  auto find_cache_block = ir::ir_utils::CollectIRNodesWithoutTensor(
      root,
      [&](const Expr* x) {
        return x->As<ir::ScheduleBlockRealize>() &&
               !x->As<ir::ScheduleBlockRealize>()->iter_values.empty() &&
               GetTensor(*x)->name == info.read_tensor->name;
      },
      true);

  PADDLE_ENFORCE_EQ(
      info.write_tensor->buffer.defined(),
      true,
      ::common::errors::InvalidArgument([&]() {
        std::ostringstream os;
        os << "[IRScheduleError] An error occurred in the schedule primitive <"
           << primitive << ">.\n"
           << "[Error info] The buffer of current write_tensor is not "
              "defined!\n"
           << "[Expr info] The Expr of current schedule is "
           << module_expr_.GetExprs() << ".";
        return os.str();
      }()));

  // Replace buffer
  auto all_tensors =
      ir::ir_utils::CollectIRNodesWithoutTensor(root, [&](const Expr* x) {
        return x->as_tensor() && x->as_tensor()->buffer.defined();
      });

  for (auto i : all_tensors) {
    if (i.as_tensor()->name != info.write_tensor->name &&
        i.as_tensor()->buffer.defined() &&
        i.as_tensor()->buffer->name == info.write_tensor->buffer->name) {
      i.as_tensor()->Bind(info.read_tensor->buffer);
    }
  }
  PADDLE_ENFORCE_EQ(
      find_cache_block.size(), 1U, ::common::errors::InvalidArgument([&]() {
        std::ostringstream os;
        os << "[IRScheduleError] An error occurred in the schedule primitive <"
           << primitive << ">.\n"
           << "[Error info] Size of find_cache_block is not 1!\n"
           << "[Expr info] The Expr of current schedule is "
           << module_expr_.GetExprs() << ".";
        return os.str();
      }()));

  return *find_cache_block.begin();
  CINN_IR_SCHEDULE_END(this->err_msg_level_);
}

void DyScheduleImpl::SyncThreads(const Expr& ir_node, bool after_node) {
  CINN_IR_SCHEDULE_BEGIN();
  std::string primitive = "SyncThreads";

  PADDLE_ENFORCE_EQ(
      ir_node.As<ScheduleBlockRealize>() || ir_node.As<ir::For>(),
      true,
      ::common::errors::InvalidArgument([&]() {
        std::ostringstream os;
        os << "[IRScheduleError] An error occurred in the schedule primitive <"
           << primitive << ">.\n"
           << "[Error info] Expr param(ir_node) should be a "
              "ScheduleBlockRealize or For!\n"
           << "[Expr info] The Expr of current schedule is "
           << module_expr_.GetExprs() << ".";
        return os.str();
      }()));

  auto root = GetRootBlock(ir_node);
  ChangeBodyToBlock::Change(&root);
  Expr sync_threads = runtime::IntrinsicCall(Void(), "__syncthreads", {});
  InsertExpr::Insert(ir_node, sync_threads, after_node, &root);
  return;
  CINN_IR_SCHEDULE_END(this->err_msg_level_);
}

void DyScheduleImpl::SetBuffer(Expr& block,  // NOLINT
                               const std::string& memory_type,
                               bool fixed) {
  CINN_IR_SCHEDULE_BEGIN();
  std::string primitive = "SetBuffer";
  PADDLE_ENFORCE_NOT_NULL(
      block.As<ir::ScheduleBlockRealize>(),
      ::common::errors::InvalidArgument([&]() {
        std::ostringstream os;
        os << "[IRScheduleError] An error occurred in the schedule primitive <"
           << primitive << ">.\n"
           << "[Error info] Expr param(block) is not a ScheduleBlockRealize!\n"
           << "[Expr info] The Expr of current schedule is "
           << module_expr_.GetExprs() << ".";
        return os.str();
      }()));

  auto find_tensor = ir::ir_utils::CollectIRNodesWithoutTensor(
      block, [&](const Expr* x) { return x->As<ir::Store>(); }, true);

  PADDLE_ENFORCE_EQ(
      find_tensor.size(), 1U, ::common::errors::InvalidArgument([&]() {
        std::ostringstream os;
        os << "[IRScheduleError] An error occurred in the schedule primitive <"
           << primitive << ">.\n"
           << "[Error info] One block should only have one Store node!(except "
              "for root block)\n"
           << "[Expr info] The Expr of current schedule is "
           << module_expr_.GetExprs() << ".";
        return os.str();
      }()));

  auto& tensor = (*find_tensor.begin()).As<ir::Store>()->tensor;
  tensor.as_tensor_ref()->WithBuffer(
      memory_type, "_" + tensor.as_tensor_ref()->name + "_temp_buffer");

  auto exprs = this->GetModule().GetExprs();
  for (auto& it_expr : exprs) {
    auto find_tensor =
        ir::ir_utils::CollectIRNodesWithoutTensor(it_expr, [&](const Expr* x) {
          return x->as_tensor() &&
                 (x->as_tensor()->name == tensor.as_tensor_ref()->name ||
                  x->as_tensor()->name ==
                      tensor.as_tensor_ref()->name + "__reduce_init");
        });
    for (auto& t : find_tensor) {
      t.as_tensor_ref()->Bind(tensor.as_tensor_ref()->buffer);
    }
  }

  // if buffer type == "local"
  if (memory_type == "local" && fixed) {
    FixLocalBufferSize mutator(block.As<ir::ScheduleBlockRealize>()
                                   ->schedule_block.As<ir::ScheduleBlock>()
                                   ->name);
    auto root = GetRootBlock(block);
    mutator(&root);
  }
  CINN_IR_SCHEDULE_END(this->err_msg_level_);
}
}  // namespace ir
}  // namespace cinn
