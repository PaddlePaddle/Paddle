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

#include "paddle/cinn/ir/group_schedule/tactic/compute_inline_tactic.h"
#include "paddle/cinn/ir/ir_analyzer/ir_analyzer.h"
#include "paddle/cinn/ir/schedule/ir_schedule_util.h"
#include "paddle/cinn/ir/utils/ir_compare.h"

namespace cinn {
namespace ir {
namespace {

/**
 * The types of the AutoInline
 */
enum class AutoInlineType : int {
  // The block cannot be inlined
  kCannotInline = 0,
  // Inline this block into the consumer
  kInlineIntoConsumer,
  // Inline this block into the producer
  kInlineIntoProducer,
};

class ComputeInlineTactic final : public ScheduleTactic {
 public:
  void Init(ScheduleContext* context) override;

  void Apply(ir::IRSchedule* sch, const std::string& block_id) override;

  std::string TacticName() const override { return "ComputeInlineTactic"; }

 private:
  AutoInlineType AnalyzeInlineType(const Expr& sche_block_realize_expr,
                                   ir::IRSchedule* ir_sch) const;
  bool CanInlineIntoConsumer(const Expr& sche_block_realize_expr,
                             ir::IRSchedule* ir_sch) const;

  // Check whether all consumers of block have their load indices aligned with
  // the block (i.e. no cross-thread access).
  bool CheckAllConsumersAligned(const Expr& block,
                                const std::vector<ir::Expr>& consumers,
                                ir::IRSchedule* ir_sch) const;

  std::unordered_set<std::string> output_names_;
  cinn::common::Target target_;
};

void ComputeInlineTactic::Init(ScheduleContext* context) {
  output_names_ = context->output_names;
  target_ = context->target;
}

int64_t GetSerialLoopExtent(const std::vector<ir::Expr>& loops) {
  int64_t extent = 1;
  for (auto& loop : loops) {
    auto* node = loop.As<ir::For>();
    if (node->is_binded()) continue;
    if (!node->extent.is_constant()) return -1;
    extent *= node->extent.as_int64();
  }
  return extent;
}

bool ComputeInlineTactic::CheckAllConsumersAligned(
    const Expr& block,
    const std::vector<ir::Expr>& consumers,
    ir::IRSchedule* ir_sch) const {
  ir::Expr simplifiedBlock = block;
  if (simplifiedBlock.is_index())
    simplifiedBlock = simplifiedBlock.as_index().Normalize();
  ir::Expr store = ir::analyzer::GetStoreOfSBlock(simplifiedBlock);
  auto* tensor = store.As<ir::Store>()->tensor.as_tensor();
  std::vector<ir::Expr> loops = ir_sch->GetLoops(simplifiedBlock);

  std::vector<ir::Expr> store_indices;
  for (ir::Expr index : store.As<ir::Store>()->indices) {
    index = ir::analyzer::ExpandIterVar(index, block);
    index = ir::analyzer::CanonicalizeLoopVar(index, loops);
    store_indices.push_back(index);
  }

  const auto CheckLoadsAligned = [&](const ir::Expr& expr) {
    bool aligned = true;
    ir::ir_utils::CollectIRNodesInOrder(expr, [&](const ir::Expr* x) {
      auto* node = x->As<ir::Load>();
      if (node && node->tensor.as_tensor()->name == tensor->name) {
        ir::ir_utils::IrEqualVisitor visitor;

        std::vector<ir::Expr> load_indices;
        for (ir::Expr index : node->indices) {
          if (index.is_index()) {
            index = index.as_index().Normalize(ir::IndexExpr::OptLevel::kLevel3);
          }
          load_indices.push_back(index);
        }
        if (load_indices != store_indices) {
          aligned = false;
        }
      }
      return false;
    });
    return aligned;
  };

  for (auto& consumer_block : consumers) {
    ir::Expr consumer_store = ir::analyzer::GetStoreOfSBlock(consumer_block);
    std::vector<ir::Expr> consumer_loops = ir_sch->GetLoops(consumer_block);
    ir::Expr value = consumer_store.As<ir::Store>()->value;
    value = ir::analyzer::ExpandIterVar(value, consumer_block);
    value = ir::analyzer::CanonicalizeLoopVar(value, consumer_loops);
    if (!CheckLoadsAligned(value))  {
      VLOG(6) << "Found store Expr " << value << ", which must be inlined into consumer due to unaligned access";
      return false;
    }
  }

  return true;
}

bool ComputeInlineTactic::CanInlineIntoConsumer(const Expr& block,
                                                ir::IRSchedule* ir_sch) const {
  ir::Expr root = ir_sch->GetRootBlock(block);
  ir::Expr store = ir::analyzer::GetStoreOfSBlock(block);
  auto* tensor = store.As<ir::Store>()->tensor.as_tensor();

  // 1. It is not a reduce nor reduce_init.
  if (ir::analyzer::IsReductionSBlock(block) || tensor->is_reduce_tensor() ||
      ir::IsReduceInitTensorName(tensor->name)) {
    return false;
  }

  // 2. It is not an output node.
  if (output_names_.count(tensor->name) > 0) {
    return false;
  }

  // 3. For block with multiple consumers, we prefer to buffer the intermediate
  //    result instead of inlining it in order to avoid redundant computation,
  //    if the following conditions are also satisfied:
  // 1) The loop extent is <= 8, otherwise the intermediate result is too large
  //    to buffer.
  // 2) Its consumers are all aligned with it, otherwise it will incur cross-
  //    thread access, which is not possible using local buffer.
  std::vector<ir::Expr> consumers = ir::GetConsumers(block, root);
  int64_t loop_extent = GetSerialLoopExtent(ir_sch->GetLoops(block));
  bool is_small_loop = loop_extent <= 8 && loop_extent != -1;
  if (consumers.size() > 1 && is_small_loop &&
      CheckAllConsumersAligned(block, consumers, ir_sch)) {
    return false;
  }

  VLOG(6) << "Found store Expr " << store << ", which CanInlineIntoConsumer";
  return true;
}

bool ContainsNodeType(ir::Expr expr,
                      const std::unordered_set<ir::IrNodeTy>& node_types) {
  std::vector<ir::Expr> collection =
      ir::ir_utils::CollectIRNodesWithoutTensor(expr, [&](const Expr* x) {
        return node_types.find(x->node_type()) != node_types.end();
      });
  return !collection.empty();
}

// TODO(Hongqing-work): IndicesToVars and AnalyzeScheduleBlockReadWriteBuffer in
// ir_analyzer.cc will cause error here, so we temporarily keep the old version
// of code from auto_schedule analyze_ir.cc and fix it later.
std::vector<ir::Var> IndicesToVars(const std::vector<ir::Expr>& indices) {
  std::vector<ir::Var> result;
  for (const ir::Expr& e : indices) {
    // Whether we have to convert other types, like const numbers to Var?
    if (e.As<ir::_Var_>() != nullptr) {
      ir::Expr copy_e = ir::ir_utils::IRCopy(e);
      ir::_Var_* var_ref = copy_e.As<ir::_Var_>();
      result.emplace_back(ir::Var(var_ref));
    }
  }
  return result;
}

void AnalyzeScheduleBlockReadWriteBuffer(ir::ScheduleBlock* sche_block) {
  if (!sche_block->read_buffers.empty() || !sche_block->write_buffers.empty()) {
    return;
  }

  ir::ir_utils::CollectIRNodesWithoutTensor(
      sche_block->body, [&](const Expr* x) {
        const ir::Load* load_expr = x->As<ir::Load>();
        if (load_expr != nullptr) {
          const ir::Tensor t = load_expr->tensor.as_tensor_ref();
          sche_block->read_buffers.emplace_back(
              ir::BufferRange(t->buffer, IndicesToVars(load_expr->indices)));
          return false;
        }
        const ir::Store* store_expr = x->As<ir::Store>();
        if (store_expr != nullptr) {
          const ir::Tensor t = store_expr->tensor.as_tensor_ref();
          sche_block->write_buffers.emplace_back(
              ir::BufferRange(t->buffer, IndicesToVars(store_expr->indices)));
          return false;
        }
        return false;
      });
}

AutoInlineType ComputeInlineTactic::AnalyzeInlineType(
    const Expr& sche_block_realize_expr, ir::IRSchedule* ir_sch) const {
  const ir::ScheduleBlockRealize* sche_block_realize =
      sche_block_realize_expr.As<ir::ScheduleBlockRealize>();
  const ir::ScheduleBlock* sche_block =
      sche_block_realize->schedule_block.As<ir::ScheduleBlock>();

  // Inline if the block has only 1 write buffer
  if (sche_block->write_buffers.size() != 1) {
    return AutoInlineType::kCannotInline;
  }

  std::unordered_set<ir::IrNodeTy> no_inline_node_types = {
      ir::IrNodeTy::IfThenElse};
  if (ContainsNodeType(sche_block->body, no_inline_node_types)) {
    return AutoInlineType::kCannotInline;
  }

  // InlineIntoConsumer other than above situations
  if (CanInlineIntoConsumer(sche_block_realize_expr, ir_sch)) {
    return AutoInlineType::kInlineIntoConsumer;
  }

  // TODO(zhhsplendid): We don't have ReverseComputeInline in IRSchedule now,
  // so we just do kInlineIntoConsumer here. Add CanInlineIntoProducer
  // once ReverseComputeInline is ready.
  return AutoInlineType::kCannotInline;
}

void ComputeInlineTactic::Apply(ir::IRSchedule* sch,
                                const std::string& block_id) {
  // TODO(LiuYang): Compute of ops will be rewritten so that we
  // don't use it in dynamic group_schedule rules temporarily.
  // if (IsProhibitScheduleExternCallBlock(node->Block())) {
  //    return;
  // }
  VLOG(6) << "try ComputeInline on: " << block_id
          << ", before ComputeInline, func body: "
          << sch->GetModule().GetExprs().front();
  ir::Expr schedule_block = sch->GetBlock(block_id);

  auto* block_realize = schedule_block.As<ir::ScheduleBlockRealize>();
  PADDLE_ENFORCE_NOT_NULL(
      block_realize,
      ::common::errors::InvalidArgument(
          "stmt is not a ScheduleBlockRealize: %s", schedule_block));

  AnalyzeScheduleBlockReadWriteBuffer(
      block_realize->schedule_block.As<ir::ScheduleBlock>());
  AutoInlineType type = AnalyzeInlineType(schedule_block, sch);

  if (type == AutoInlineType::kInlineIntoConsumer) {
    VLOG(6) << "Apply ComputeInline on " << schedule_block;
    sch->ComputeInline(schedule_block);
    VLOG(6) << "After ComputeInline: " << schedule_block;

  } else if (type == AutoInlineType::kInlineIntoProducer) {
    // TODO(zhhsplendid): We don't have ReverseComputeInline in IRSchedule now,
    // so we just do kInlineIntoConsumer here. Add CanInlineIntoConsumer
    // once ReverseComputeInline is ready.
  }

  // Make sure re-apply the AutoInline won't be error.
  // AutoInline changes the read and write buffers of schedule blocks,
  // we need to re-analyze
  auto all_block_realizes = sch->GetAllBlocks();
  for (size_t i = 0; i < all_block_realizes.size(); ++i) {
    ir::ScheduleBlockRealize* sche_block_realize =
        all_block_realizes[i].As<ir::ScheduleBlockRealize>();
    ir::ScheduleBlock* sche_block =
        sche_block_realize->schedule_block.As<ir::ScheduleBlock>();
    sche_block->read_buffers = {};
    sche_block->write_buffers = {};
    AnalyzeScheduleBlockReadWriteBuffer(sche_block);
  }
  VLOG(6) << "try ComputeInline on: " << block_id
          << ", after ComputeInline, func body: "
          << sch->GetModule().GetExprs().front();
}

}  // namespace

std::unique_ptr<ScheduleTactic> CreateComputeInlineTactic() {
  return std::make_unique<ComputeInlineTactic>();
}

}  // namespace ir
}  // namespace cinn
