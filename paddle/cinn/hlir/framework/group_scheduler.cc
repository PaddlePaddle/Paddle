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

#include "paddle/cinn/hlir/framework/group_scheduler.h"
#include "paddle/cinn/auto_schedule/search_space/auto_gen_rule/auto_inline.h"
#include "paddle/cinn/common/cas.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/ir/schedule/ir_schedule_util.h"
#include "paddle/cinn/ir/tensor.h"
#include "paddle/cinn/ir/utils/ir_copy.h"
#include "paddle/cinn/ir/utils/ir_nodes_collector.h"
#include "paddle/cinn/ir/utils/ir_printer.h"
#include "paddle/cinn/optim/replace_var_with_expr.h"

namespace cinn {
namespace hlir {
namespace framework {

static const std::unordered_set<std::string>
    kProhibitScheduleExternalFuncNames = {
#define CINN_NVGPU_FUNC2STRING(str) #str
#define CINN_NVGPU_FUNC_TYPE(FUNC, TYPE) \
  CINN_NVGPU_FUNC2STRING(cinn_nvgpu_##FUNC##TYPE)

#define GEN_FUNC_NAME(_, impl) \
  _(impl, gt_num)              \
  _(impl, lt_num)              \
  _(impl, index_add)           \
  _(impl, next_smallest)

#define GEN_FUNC_NAME_WITH_TYPE(_, ...)                                     \
  _(__VA_ARGS__, _bool), _(__VA_ARGS__, _fp16), _(__VA_ARGS__, _fp32),      \
      _(__VA_ARGS__, _fp64), _(__VA_ARGS__, _uint8), _(__VA_ARGS__, _int8), \
      _(__VA_ARGS__, _int16), _(__VA_ARGS__, _int32), _(__VA_ARGS__, _int64),

        GEN_FUNC_NAME(GEN_FUNC_NAME_WITH_TYPE, CINN_NVGPU_FUNC_TYPE)
#undef GEN_FUNC_NAME
};

bool IsProhibitScheduleExternCallBlock(ir::Expr block) {
  ir::ScheduleBlockRealize* sch_block_realize =
      block.As<ir::ScheduleBlockRealize>();
  CHECK_NOTNULL(sch_block_realize);
  ir::ScheduleBlock* sch_block =
      sch_block_realize->schedule_block.As<ir::ScheduleBlock>();
  CHECK_NOTNULL(sch_block);

  auto find_call = ir::CollectIRNodesWithoutTensor(
      sch_block->body, [&](const Expr* x) { return x->As<ir::Call>(); });
  for (ir::Expr call : find_call) {
    ir::Call* call_node = call.As<ir::Call>();
    if (call.As<ir::Call>() && kProhibitScheduleExternalFuncNames.count(
                                   call.As<ir::Call>()->name) != 0) {
      return true;
    }
  }
  return false;
}

// Find loops with same extents of 2 ScheduleBlock
std::vector<std::tuple<ir::Expr, ir::Expr>> FindSameOuterLoops(
    ir::ScheduleBlockNode* source_node, ir::ScheduleBlockNode* target_node) {
  std::vector<ir::Expr> src_ctrl_stmts = source_node->ControlStmts();
  std::vector<ir::Expr> tgt_ctrl_stmts = target_node->ControlStmts();
  std::vector<std::tuple<ir::Expr, ir::Expr>> same_loops;
  int min_stmt_size = std::min(src_ctrl_stmts.size(), tgt_ctrl_stmts.size());
  for (int i = 0; i < min_stmt_size; ++i) {
    if (src_ctrl_stmts[i].As<ir::For>() && tgt_ctrl_stmts[i].As<ir::For>() &&
        ir::GetLoopExtent(src_ctrl_stmts[i]) ==
            GetLoopExtent(tgt_ctrl_stmts[i])) {
      same_loops.push_back(
          std::make_tuple(src_ctrl_stmts[i], tgt_ctrl_stmts[i]));
    } else {
      break;
    }
  }

  return same_loops;
}

GroupScheduler::GroupScheduler(ir::IRSchedule* ir_sch,
                               const std::shared_ptr<Graph::Group>& group,
                               const common::Target& target)
    : ir_sch_(ir_sch), group_(group), target_(target) {
  schedule_block_graph_ = std::make_unique<ir::ScheduleBlockGraph>(*ir_sch_);
}

void GroupScheduler::operator()() {
  feasible_conditions_.emplace_back(&GroupScheduler::IsKeepGraphDependency);
  DoLoopAlignment();
  DoComputeInline();
  DoHorizontalLoopFusion();
  DoVerticalLoopFusion();
  AllocateStorage();
}

NodePriority GroupScheduler::CalculateNodePriority(
    const ir::ScheduleBlockNode* node) const {
  bool has_loop_binded = false;
  int64_t score = 1;
  for (Expr expr : node->ControlStmts()) {
    ir::For* for_node = expr.As<ir::For>();
    if (for_node != nullptr) {
      score *= ir::GetLoopExtent(expr);
    }
    if (for_node->is_binded()) {
      has_loop_binded = true;
    }
  }
  score += node->ControlStmts().size();

  VLOG(6) << "The priority score of node " << node->id() << " is " << score;
  VLOG(6) << "The node has_loop_binded: " << has_loop_binded;
  return NodePriority{has_loop_binded, score};
}

ir::ScheduleBlockNode* GroupScheduler::FindGlobalMasterNode() const {
  NodePriority max{false, std::numeric_limits<int64_t>::min()};
  ir::ScheduleBlockNode* master = nullptr;
  auto FindMaster = [&](ir::ScheduleBlockNode* node) {
    NodePriority priority = CalculateNodePriority(node);
    VLOG(6) << "The priority score of node " << node->id() << " is "
            << priority.score
            << ", has_loop_binded: " << priority.has_loop_binded;
    if (max < priority) {
      max = priority;
      master = node;
    }
  };

  schedule_block_graph_->NodesWalk(FindMaster);
  CHECK(master) << "Cannot find global master node";
  VLOG(6) << "Find the global master node: " << master->id();
  return master;
}

std::unordered_set<std::string> GroupScheduler::OutputTensorNames() const {
  std::unordered_set<std::string> output_tensor_names;
  std::transform(
      group_->output_nodes.begin(),
      group_->output_nodes.end(),
      std::inserter(output_tensor_names, output_tensor_names.begin()),
      [](const Node* node) {
        NodeData* node_data =
            (*node->outlinks().begin())->sink()->safe_as<NodeData>();
        CHECK(node_data);
        return node_data->id();
      });

  for (ir::ScheduleBlockNode* node : schedule_block_graph_->EndPoints()) {
    output_tensor_names.insert(node->id());
  }
  return output_tensor_names;
}

void GroupScheduler::DoLoopAlignment() {
  VLOG(5) << "[Start LoopAlignment] func body: "
          << ir_sch_->GetModule().GetExprs().front();
  ir::ScheduleBlockNode* global_master = FindGlobalMasterNode();

  auto LoopAlignmentFunc = [&](ir::ScheduleBlockNode* node) {
    VLOG(6) << "try to align loops of block: " << node->id()
            << " with block: " << global_master->id();
    if (node == global_master) {
      return false;
    }

    for (ir::Expr expr : node->ControlStmts()) {
      if (expr.As<ir::For>() != nullptr &&
          (expr.As<ir::For>()->for_type() == ir::ForType::GPUBlock ||
           expr.As<ir::For>()->for_type() == ir::ForType::GPUThread)) {
        return false;
      }
    }

    // 1. Fuse source loops
    ir::Expr source_loop = ir_sch_->Fuse(node->ControlStmts());
    int total_source_extent = ir::GetLoopExtent(source_loop);

    // 2. Split source loop to align with the target loops
    std::vector<int> factors;
    int cur_extent = 1;
    std::vector<ir::Expr> target_loops = global_master->ControlStmts();
    for (ir::Expr target_loop : target_loops) {
      if (target_loop.As<ir::For>() == nullptr) {
        continue;
      }
      if (cur_extent >= total_source_extent && factors.size() &&
              ir::GetLoopExtent(target_loop) > 1 ||
          total_source_extent % (cur_extent * ir::GetLoopExtent(target_loop)) !=
              0) {
        break;
      }
      factors.push_back(ir::GetLoopExtent(target_loop));
      cur_extent *= ir::GetLoopExtent(target_loop);
    }
    if (cur_extent < total_source_extent) {
      factors.push_back(-1);
    }
    std::vector<ir::Expr> source_loops;
    if (factors.size() > 0 && factors[0] < total_source_extent) {
      source_loops = ir_sch_->Split(source_loop, factors);
    } else {
      source_loops = {source_loop};
    }

    // 3. Copy bind info from target loops
    for (int idx = 0; idx < target_loops.size(); ++idx) {
      std::string thread_axis = "";
      ir::ForType target_for_type = target_loops[idx].As<ir::For>()->for_type();
      if (target_for_type == ir::ForType::GPUBlock) {
        thread_axis += "blockIdx.";
      } else if (target_for_type == ir::ForType::GPUThread) {
        thread_axis += "threadIdx.";
      } else {
        continue;
      }
      int offset = target_loops[idx].As<ir::For>()->bind_info().offset;
      thread_axis += ('x' + offset);
      if (idx >= source_loops.size()) {
        ir::Expr unit_loop = ir_sch_->AddUnitLoop(node->Block());
        ir_sch_->Bind(unit_loop, thread_axis);
      } else {
        ir_sch_->Bind(source_loops[idx], thread_axis);
      }
    }
    return true;
  };

  schedule_block_graph_->DFSTopoWalk(LoopAlignmentFunc);
  VLOG(5) << "[After LoopAlignment] func body: "
          << ir_sch_->GetModule().GetExprs().front();
}

void GroupScheduler::DoComputeInline() {
  VLOG(5) << "[Start DoComputeInline] func body: "
          << ir_sch_->GetModule().GetExprs().front();

  std::unordered_set<std::string> no_inline_output_names = OutputTensorNames();
  auto_schedule::AutoInline inliner(target_, no_inline_output_names);

  auto InlineFunc = [&](ir::ScheduleBlockNode* node) {
    if (IsProhibitScheduleExternCallBlock(node->Block())) {
      return;
    }
    VLOG(6) << "try ComputeInline on: " << node->id()
            << ", before ComputeInline, func body: "
            << ir_sch_->GetModule().GetExprs().front();
    ir::Expr schedule_block = node->Block();
    inliner.Apply(ir_sch_, schedule_block);
    VLOG(6) << "try ComputeInline on: " << node->id()
            << ", after ComputeInline, func body: "
            << ir_sch_->GetModule().GetExprs().front();
  };

  schedule_block_graph_->DFSTopoWalk(InlineFunc);
  schedule_block_graph_->Update(*ir_sch_);
  VLOG(5) << "[After DoComputeInline] func body: "
          << ir_sch_->GetModule().GetExprs().front();
}

void GroupScheduler::DoHorizontalLoopFusion() {
  VLOG(5) << "[Start DoHorizontalLoopFusion] func body: "
          << ir_sch_->GetModule().GetExprs().front();

  std::vector<ir::ScheduleBlockNode*> end_nodes =
      schedule_block_graph_->EndPoints();
  std::reverse(end_nodes.begin(), end_nodes.end());
  ir::ScheduleBlockNode* master_node = end_nodes.front();
  CHECK_NOTNULL(master_node);
  for (int i = 1; i < end_nodes.size(); ++i) {
    if (IsProhibitScheduleExternCallBlock(end_nodes[i]->Block())) {
      continue;
    }
    VLOG(6) << "try to fuse loop of " << end_nodes[i]->id() << " to "
            << master_node->id();
    std::vector<std::tuple<cinn::ir::Expr, cinn::ir::Expr>>&& same_loops =
        FindSameOuterLoops(end_nodes[i], master_node);
    if (same_loops.size() == 0) {
      continue;
    }
    ir::Expr target_loop = std::get<1>(same_loops.back());
    VLOG(6) << "target_loop: " << target_loop;
    ir_sch_->SimpleComputeAt(end_nodes[i]->Block(), target_loop);
    VLOG(6) << "after fuse: " << ir_sch_->GetModule().GetExprs().front();
  }

  VLOG(5) << "[After DoHorizontalLoopFusion] func body: "
          << ir_sch_->GetModule().GetExprs().front();
}

void GroupScheduler::DoVerticalLoopFusion() {
  VLOG(5) << "[Start DoVerticalLoopFusion] func body: "
          << ir_sch_->GetModule().GetExprs().front();
  UpdateBlockOrder();

  auto FindMaster =
      [&](ir::ScheduleBlockNode* node) -> std::vector<ir::ScheduleBlockNode*> {
    std::vector<ir::ScheduleBlockNode*> masters = node->Consumers();
    std::sort(
        masters.begin(),
        masters.end(),
        [&](const ir::ScheduleBlockNode* a, const ir::ScheduleBlockNode* b) {
          return this->CalculateNodePriority(b) <
                 this->CalculateNodePriority(a);
        });
    return masters;
  };

  auto ComputeAtFunc = [&](ir::ScheduleBlockNode* node) {
    if (IsProhibitScheduleExternCallBlock(node->Block())) {
      return;
    }
    std::vector<ir::ScheduleBlockNode*> masters = FindMaster(node);
    if (masters.size() == 0) {
      return;
    }
    ir::Expr target_loop;
    bool find_target_loop = false;
    // Collect infomation of original loops
    std::vector<ir::Expr> original_ctrl_stmts = node->ControlStmts();
    int64_t original_total_loop_extent = 1;
    std::vector<std::pair<std::string, int>> original_loop_infos;
    std::unordered_set<ir::IrNode*> original_loop_node_ptrs;
    for (ir::Expr stmt : original_ctrl_stmts) {
      if (stmt.As<ir::For>()) {
        int extent = ir::GetLoopExtent(stmt);
        original_total_loop_extent *= extent;
        std::string thread_axis = "";
        ir::ForType target_for_type = stmt.As<ir::For>()->for_type();
        if (target_for_type == ir::ForType::GPUBlock) {
          thread_axis += "blockIdx.";
        } else if (target_for_type == ir::ForType::GPUThread) {
          thread_axis += "threadIdx.";
        } else {
          continue;
        }
        int offset = stmt.As<ir::For>()->bind_info().offset;
        thread_axis += ('x' + offset);
        original_loop_infos.push_back(std::make_pair(thread_axis, extent));
        original_loop_node_ptrs.insert(stmt.ptr());
      }
    }

    for (ir::ScheduleBlockNode* master : masters) {
      // Find the target loop candidates;
      std::vector<ir::Expr> target_loop_candidates;
      int64_t total_loop_extent = 1;
      std::vector<std::tuple<cinn::ir::Expr, cinn::ir::Expr>> same_loops =
          FindSameOuterLoops(node, master);
      for (const std::tuple<cinn::ir::Expr, cinn::ir::Expr>& same_loop :
           same_loops) {
        if (std::get<0>(same_loop).ptr() != std::get<1>(same_loop).ptr()) {
          target_loop_candidates.push_back(std::get<1>(same_loop));
        }
      }
      // Find the target loop with the highest priority and passing the
      // feasibility condition check
      for (std::vector<ir::Expr>::reverse_iterator iter =
               target_loop_candidates.rbegin();
           iter != target_loop_candidates.rend();
           ++iter) {
        ir::Expr candidate_loop = *iter;
        if (candidate_loop.As<ir::For>() &&
            this->MeetConditions(node->Block(), candidate_loop, 0)) {
          target_loop = candidate_loop;
          find_target_loop = true;
          break;
        }
      }
      if (find_target_loop) {
        VLOG(6) << "try to fuse loop of " << node->id() << " to "
                << master->id();
        break;
      }
    }

    // Do schedule
    if (find_target_loop) {
      ir_sch_->SimpleComputeAt(node->Block(), target_loop);
      VLOG(6) << "after compute at: " << ir_sch_->GetModule().GetExprs()[0];
      std::vector<ir::Expr> new_stmts = node->ControlStmts();
      for (int idx = 0; idx < original_loop_infos.size(); ++idx) {
        if (idx < new_stmts.size()) {
          CHECK(new_stmts[idx].As<ir::For>());
          if (new_stmts[idx].As<ir::For>()->is_serial()) {
            ir_sch_->Bind(new_stmts[idx], original_loop_infos[idx].first);
          }
        } else {
          ir::Expr unit_loop = ir_sch_->AddUnitLoop(node->Block());
          ir_sch_->Bind(unit_loop, original_loop_infos[idx].first);
        }
      }
      VLOG(6) << "after loop info copy: " << ir_sch_->GetModule().GetExprs()[0];
      // Update block and control stmts order after schedule.
      this->UpdateBlockOrder();
    } else {
      LOG(INFO) << "Cannot find a loop of masters to ComputeAt, do not merge.\n"
                << "The schedule block: " << node->Block();
    }
  };

  schedule_block_graph_->DFSTopoWalk(ComputeAtFunc);
  VLOG(5) << "[After DoVerticalLoopFusion] func body: "
          << ir_sch_->GetModule().GetExprs().front();
}

void GroupScheduler::AllocateStorage() {
  VLOG(5) << "[Start AllocateStorage] func body: "
          << ir_sch_->GetModule().GetExprs().front();

  // Record ir::For using index structure: <block_name, <var_name, for_node>>
  std::unordered_map<std::string, std::unordered_map<std::string, ir::Expr>>
      for_map;
  std::unordered_set<std::string> sync_mark;

  // function to update for_map
  auto UpdateVarNameToForMap = [&](ir::Expr root) {
    std::vector<ir::Expr> all_blocks = ir_sch_->GetAllBlocks();
    for (const ir::Expr& block : all_blocks) {
      std::string block_name = block.As<ir::ScheduleBlockRealize>()
                                   ->schedule_block.As<ir::ScheduleBlock>()
                                   ->name;
      std::vector<ir::Expr> for_expr = ir_sch_->GetLoops(block);
      for (ir::Expr for_expr : for_expr) {
        for_map[block_name][for_expr.As<ir::For>()->loop_var->name] = for_expr;
        VLOG(6) << "for_map.insert: <" << block_name << ", "
                << for_expr.As<ir::For>()->loop_var->name << ">";
      }
    }
  };

  // function to analyze and flatten indices to one dim of load_or_store node
  auto AnalyzeIndiceValue = [](ir::Expr load_or_store,
                               ir::Expr block) -> ir::Expr {
    std::vector<ir::Expr> indices;
    ir::Tensor tensor;
    if (load_or_store.As<ir::Load>()) {
      indices = load_or_store.As<ir::Load>()->indices;
      tensor = load_or_store.As<ir::Load>()->tensor.as_tensor_ref();
    } else {
      indices = load_or_store.As<ir::Store>()->indices;
      tensor = load_or_store.As<ir::Store>()->tensor.as_tensor_ref();
    }
    std::vector<ir::Var> iter_vars =
        block.As<ir::ScheduleBlockRealize>()
            ->schedule_block.As<ir::ScheduleBlock>()
            ->iter_vars;
    std::vector<ir::Expr> iter_values =
        block.As<ir::ScheduleBlockRealize>()->iter_values;
    struct VarHash {
      size_t operator()(const ir::Var& var) const {
        std::string name = var->name;
        return std::hash<std::string>()(name);
      }
    };
    std::vector<int> strides;
    int extent = 1;
    for (int idx = tensor->shape.size() - 1; idx >= 0; --idx) {
      strides.insert(strides.begin(), extent);
      tensor->shape[idx] = common::AutoSimplify(tensor->shape[idx]);
      CHECK(tensor->shape[idx].is_constant())
          << "Shape of tensor: " << tensor << " is not constant";
      extent *= tensor->shape[idx].get_constant();
    }
    ir::Expr flatten_indice(0);
    for (int idx = 0; idx < indices.size(); ++idx) {
      flatten_indice = flatten_indice + ir::Expr(strides[idx]) * indices[idx];
    }
    flatten_indice = common::AutoSimplify(flatten_indice);
    for (int idx = 0; idx < iter_vars.size(); ++idx) {
      optim::ReplaceVarWithExpr(
          &flatten_indice, iter_vars[idx], iter_values[idx]);
    }
    flatten_indice = common::AutoSimplify(flatten_indice);
    VLOG(6) << "flatten_indice of " << load_or_store << " : " << flatten_indice;
    return flatten_indice;
  };

  enum class CudaBindInfo : int {
    kCudaBlock,
    kCudaThread,
    kSerial,
    kCudaThreadAndSerial,
  };

  struct Range {
    int min;
    int max;
  };

  // function to calculate the range of the specified CUDA axis in a indice
  // expression
  auto CalculateRange = [&for_map](ir::Expr indice_value,
                                   const CudaBindInfo& bind_info,
                                   const std::string& block_name) {
    ir::Expr copy_for_upper_bound = optim::IRCopy(indice_value);
    ir::Expr copy_for_lower_bound = optim::IRCopy(indice_value);
    std::set<ir::Expr> var_set = ir::CollectIRNodesWithoutTensor(
        indice_value, [](const ir::Expr* x) { return x->as_var(); });
    for (ir::Expr var : var_set) {
      std::string name = var.as_var_ref()->name;
      CHECK(for_map.find(block_name) != for_map.end());
      CHECK(for_map[block_name].find(name) != for_map[block_name].end());
      ir::Expr for_expr = for_map[block_name][name];
      if (bind_info == CudaBindInfo::kCudaBlock) {
        if (for_expr.As<ir::For>()->is_gpu_block_binded()) {
          optim::ReplaceVarWithExpr(&copy_for_upper_bound,
                                    var.as_var_ref(),
                                    for_expr.As<ir::For>()->min +
                                        for_expr.As<ir::For>()->extent -
                                        Expr(1));
          optim::ReplaceVarWithExpr(&copy_for_lower_bound,
                                    var.as_var_ref(),
                                    for_expr.As<ir::For>()->min);
        } else {
          optim::ReplaceVarWithExpr(
              &copy_for_upper_bound, var.as_var_ref(), ir::Expr(0));
          optim::ReplaceVarWithExpr(
              &copy_for_lower_bound, var.as_var_ref(), ir::Expr(0));
        }
      } else if (bind_info == CudaBindInfo::kCudaThread) {
        if (for_expr.As<ir::For>()->is_gpu_thread_binded()) {
          optim::ReplaceVarWithExpr(&copy_for_upper_bound,
                                    var.as_var_ref(),
                                    for_expr.As<ir::For>()->min +
                                        for_expr.As<ir::For>()->extent -
                                        Expr(1));
          optim::ReplaceVarWithExpr(&copy_for_lower_bound,
                                    var.as_var_ref(),
                                    for_expr.As<ir::For>()->min);
        } else {
          optim::ReplaceVarWithExpr(
              &copy_for_upper_bound, var.as_var_ref(), ir::Expr(0));
          optim::ReplaceVarWithExpr(
              &copy_for_lower_bound, var.as_var_ref(), ir::Expr(0));
        }
      } else if (bind_info == CudaBindInfo::kSerial) {
        if (!for_expr.As<ir::For>()->is_gpu_thread_binded() &&
            !for_expr.As<ir::For>()->is_gpu_block_binded()) {
          optim::ReplaceVarWithExpr(&copy_for_upper_bound,
                                    var.as_var_ref(),
                                    for_expr.As<ir::For>()->min +
                                        for_expr.As<ir::For>()->extent -
                                        Expr(1));
          optim::ReplaceVarWithExpr(&copy_for_lower_bound,
                                    var.as_var_ref(),
                                    for_expr.As<ir::For>()->min);
        } else {
          optim::ReplaceVarWithExpr(
              &copy_for_upper_bound, var.as_var_ref(), ir::Expr(0));
          optim::ReplaceVarWithExpr(
              &copy_for_lower_bound, var.as_var_ref(), ir::Expr(0));
        }
      } else if (bind_info == CudaBindInfo::kCudaThreadAndSerial) {
        if (!for_expr.As<ir::For>()->is_gpu_block_binded()) {
          optim::ReplaceVarWithExpr(&copy_for_upper_bound,
                                    var.as_var_ref(),
                                    for_expr.As<ir::For>()->min +
                                        for_expr.As<ir::For>()->extent -
                                        Expr(1));
          optim::ReplaceVarWithExpr(&copy_for_lower_bound,
                                    var.as_var_ref(),
                                    for_expr.As<ir::For>()->min);
        } else {
          optim::ReplaceVarWithExpr(
              &copy_for_upper_bound, var.as_var_ref(), ir::Expr(0));
          optim::ReplaceVarWithExpr(
              &copy_for_lower_bound, var.as_var_ref(), ir::Expr(0));
        }
      }
    }
    VLOG(6) << "lower_bound before simplify of " << indice_value << " = "
            << copy_for_lower_bound;
    copy_for_lower_bound = common::AutoSimplify(copy_for_lower_bound);
    VLOG(6) << "upper_bound before simplify of " << indice_value << " = "
            << copy_for_upper_bound;
    copy_for_upper_bound = common::AutoSimplify(copy_for_upper_bound);
    VLOG(6) << "lower_bound of " << indice_value << " = "
            << copy_for_lower_bound;
    VLOG(6) << "upper_bound of " << indice_value << " = "
            << copy_for_upper_bound;
    return Range{static_cast<int>(copy_for_lower_bound.get_constant()),
                 static_cast<int>(copy_for_upper_bound.get_constant())};
  };

  // function to calculate the coefficient and range of the specified for_type
  // in a indice expression
  auto GetCoefficientAndRange = [&for_map](ir::Expr indice_value,
                                           const ir::ForType& for_type,
                                           const std::string& block_name) {
    ir::Expr indice_copy = optim::IRCopy(indice_value);
    Range range{1, 1};
    std::set<ir::Expr> var_set = ir::CollectIRNodesWithoutTensor(
        indice_copy, [](const ir::Expr* x) { return x->as_var(); });
    for (ir::Expr var : var_set) {
      std::string name = var.as_var_ref()->name;
      CHECK(for_map.find(block_name) != for_map.end());
      CHECK(for_map[block_name].find(name) != for_map[block_name].end());
      ir::Expr for_expr = for_map[block_name][name];
      if (for_type == for_expr.As<ir::For>()->for_type() &&
          for_expr.As<ir::For>()->extent.get_constant() > 1) {
        optim::ReplaceVarWithExpr(&indice_copy, var.as_var_ref(), ir::Expr(1));
        range.min *= for_expr.As<ir::For>()->min.get_constant();
        range.max *= for_expr.As<ir::For>()->min.get_constant() +
                     for_expr.As<ir::For>()->extent.get_constant();
      } else {
        optim::ReplaceVarWithExpr(&indice_copy, var.as_var_ref(), ir::Expr(0));
      }
    }
    VLOG(6) << "before simplify, the coefficient of " << indice_value << " = "
            << indice_copy << ", range = (" << range.min << ", " << range.max
            << ")";
    indice_copy = common::AutoSimplify(indice_copy);
    VLOG(6) << "after simplify, the coefficient of " << indice_value << " = "
            << indice_copy << ", range = (" << range.min << ", " << range.max
            << ")";
    return std::make_pair(static_cast<int>(indice_copy.get_constant()), range);
  };

  // Determine whether the indice of a pair of Store and Load cross CUDA threads
  auto IsCrossThread = [&](ir::Expr store_indice_value,
                           ir::Expr load_indice_value,
                           const std::string& store_block_name,
                           const std::string& load_block_name) {
    Range store_thread_overall_range = CalculateRange(
        store_indice_value, CudaBindInfo::kCudaThread, store_block_name);
    Range load_thread_overall_range = CalculateRange(
        load_indice_value, CudaBindInfo::kCudaThread, load_block_name);
    Range store_serial_overall_range = CalculateRange(
        store_indice_value, CudaBindInfo::kSerial, store_block_name);
    Range load_serial_overall_range = CalculateRange(
        load_indice_value, CudaBindInfo::kSerial, load_block_name);
    auto store_thread_coefficient_and_range = GetCoefficientAndRange(
        store_indice_value, ir::ForType::GPUThread, store_block_name);
    auto load_thread_coefficient_and_range = GetCoefficientAndRange(
        load_indice_value, ir::ForType::GPUThread, load_block_name);
    VLOG(6) << "store_thread_overall_range = ("
            << store_thread_overall_range.min << ", "
            << store_thread_overall_range.max << ")";
    VLOG(6) << "load_thread_overall_range = (" << load_thread_overall_range.min
            << ", " << load_thread_overall_range.max << ")";
    VLOG(6) << "store_serial_overall_range = ("
            << store_serial_overall_range.min << ", "
            << store_serial_overall_range.max << ")";
    VLOG(6) << "load_serial_overall_range = (" << load_serial_overall_range.min
            << ", " << load_serial_overall_range.max << ")";
    VLOG(6) << "store_thread_coefficient_and_range = <"
            << store_thread_coefficient_and_range.first << ", ("
            << store_thread_coefficient_and_range.second.min << ", "
            << store_thread_coefficient_and_range.second.max << ")>";
    VLOG(6) << "load_thread_coefficient_and_range = <"
            << load_thread_coefficient_and_range.first << ", ("
            << load_thread_coefficient_and_range.second.min << ", "
            << load_thread_coefficient_and_range.second.max << ")>";
    return !(store_thread_overall_range.min <= load_thread_overall_range.min &&
             store_thread_overall_range.max >= load_thread_overall_range.max &&
             store_serial_overall_range.min <= load_serial_overall_range.min &&
             store_serial_overall_range.max >= load_serial_overall_range.max &&
             (store_thread_coefficient_and_range.first ==
                  load_thread_coefficient_and_range.first ||
              load_thread_coefficient_and_range.first == 0) &&
             store_thread_coefficient_and_range.second.min <=
                 load_thread_coefficient_and_range.second.min &&
             store_thread_coefficient_and_range.second.max >=
                 load_thread_coefficient_and_range.second.max);
  };

  // Determine whether the indice of a pair of Store and Load cross CUDA block
  auto IsCrossBlock = [&](ir::Expr store_indice_value,
                          ir::Expr load_indice_value,
                          const std::string& store_block_name,
                          const std::string& load_block_name) {
    Range store_block_overall_range = CalculateRange(
        store_indice_value, CudaBindInfo::kCudaBlock, store_block_name);
    Range load_block_overall_range = CalculateRange(
        load_indice_value, CudaBindInfo::kCudaBlock, load_block_name);
    Range store_thread_and_serial_overall_range =
        CalculateRange(store_indice_value,
                       CudaBindInfo::kCudaThreadAndSerial,
                       store_block_name);
    Range load_thread_and_serial_overall_range = CalculateRange(
        load_indice_value, CudaBindInfo::kCudaThreadAndSerial, load_block_name);
    auto store_block_coefficient_and_range = GetCoefficientAndRange(
        store_indice_value, ir::ForType::GPUBlock, store_block_name);
    auto load_block_coefficient_and_range = GetCoefficientAndRange(
        load_indice_value, ir::ForType::GPUBlock, load_block_name);
    VLOG(6) << "store_block_overall_range = (" << store_block_overall_range.min
            << ", " << store_block_overall_range.max << ")";
    VLOG(6) << "load_block_overall_range = (" << load_block_overall_range.min
            << ", " << load_block_overall_range.max << ")";
    VLOG(6) << "store_thread_and_serial_overall_range = ("
            << store_thread_and_serial_overall_range.min << ", "
            << store_thread_and_serial_overall_range.max << ")";
    VLOG(6) << "load_thread_and_serial_overall_range = ("
            << load_thread_and_serial_overall_range.min << ", "
            << load_thread_and_serial_overall_range.max << ")";
    VLOG(6) << "store_block_coefficient_and_range = <"
            << store_block_coefficient_and_range.first << ", ("
            << store_block_coefficient_and_range.second.min << ", "
            << store_block_coefficient_and_range.second.max << ")>";
    VLOG(6) << "load_block_coefficient_and_range = <"
            << load_block_coefficient_and_range.first << ", ("
            << load_block_coefficient_and_range.second.min << ", "
            << load_block_coefficient_and_range.second.max << ")>";
    return !(store_block_overall_range.min <= load_block_overall_range.min &&
             store_block_overall_range.max >= load_block_overall_range.max &&
             store_thread_and_serial_overall_range.min <=
                 load_thread_and_serial_overall_range.min &&
             store_thread_and_serial_overall_range.max >=
                 load_thread_and_serial_overall_range.max &&
             (store_block_coefficient_and_range.first ==
                  load_block_coefficient_and_range.first ||
              load_block_coefficient_and_range.first == 0) &&
             store_block_coefficient_and_range.second.min <=
                 load_block_coefficient_and_range.second.min &&
             store_block_coefficient_and_range.second.max >=
                 load_block_coefficient_and_range.second.max);
  };

  // function to set storage of each tensor
  auto SetStorage = [&](ir::ScheduleBlockNode* node) {
    if (IsProhibitScheduleExternCallBlock(node->Block())) {
      return;
    }
    ir::MemoryType memory_type = ir::MemoryType::GPULocal;
    ir::Expr cur_block = node->Block();
    ir::Expr root_block = ir_sch_->GetRootBlock(cur_block);
    UpdateVarNameToForMap(root_block);
    std::vector<ir::Expr> consumer_blocks =
        ir::GetConsumers(cur_block, root_block);
    // find store and corresponding load nodes
    ir::Expr find_store =
        *ir::CollectIRNodesWithoutTensor(
             cur_block,
             [&](const ir::Expr* x) { return x->As<ir::Store>(); },
             true)
             .begin();
    ir::Expr store_indice_value = AnalyzeIndiceValue(find_store, cur_block);
    std::vector<std::tuple<ir::Expr, ir::Expr>> loads_and_blocks;
    for (const ir::Expr& consumer_block : consumer_blocks) {
      ir::CollectIRNodesWithoutTensor(consumer_block, [&](const Expr* x) {
        if (x->As<ir::Load>() &&
            (x->As<ir::Load>()->name() == find_store.As<ir::Store>()->name())) {
          loads_and_blocks.push_back(std::make_tuple(*x, consumer_block));
        }
        return false;
      });
    }
    // Traverse load nodes to check if there are loads that cross cuda blocks or
    // threads
    for (const auto& load_and_block : loads_and_blocks) {
      ir::Expr load = std::get<0>(load_and_block);
      ir::Expr consumer_block = std::get<1>(load_and_block);
      std::string consumer_block_name =
          consumer_block.As<ir::ScheduleBlockRealize>()
              ->schedule_block.As<ir::ScheduleBlock>()
              ->name;
      ir::Expr load_indice_value = AnalyzeIndiceValue(load, consumer_block);
      if (IsCrossBlock(store_indice_value,
                       load_indice_value,
                       node->id(),
                       consumer_block_name)) {
        memory_type = ir::MemoryType::Auto;
        break;
      } else if (IsCrossThread(store_indice_value,
                               load_indice_value,
                               node->id(),
                               consumer_block_name)) {
        memory_type = ir::MemoryType::GPUShared;
      }
    }
    // Set output node to global
    std::unordered_set<std::string> output_names = OutputTensorNames();
    if (output_names.count(node->id()) > 0) {
      memory_type = ir::MemoryType::Auto;
    }
    // Set the reduce_init tensor and the real tensor to the same memory
    if (ir::IsReduceInitTensorName(node->id())) {
      ir::Expr block =
          ir_sch_->GetBlock(ir::GetOriginalReduceTensorName(node->id()));
      memory_type = ir::GetTensor(block)->buffer->memory_type;
    }
    // Do schedule
    if (memory_type == ir::MemoryType::Auto) {
      VLOG(6) << "Set store tensor of block " << node->id() << " to global";
    } else if (memory_type == ir::MemoryType::GPUShared) {
      VLOG(6) << "Set store tensor of block " << node->id() << " to shared";
      ir_sch_->SetBuffer(cur_block, "shared");
      std::vector<ir::Expr> loops = ir_sch_->GetLoops(cur_block);
      if (sync_mark.count(ir::GetOriginalReduceTensorName(node->id())) == 0) {
        for (int i = loops.size() - 1; i >= 0; --i) {
          if (loops[i].As<ir::For>()->is_binded()) {
            ir_sch_->SyncThreads(loops[i], true);
            break;
          }
        }
        sync_mark.insert(ir::GetOriginalReduceTensorName(node->id()));
      }
    } else if (memory_type == ir::MemoryType::GPULocal) {
      VLOG(6) << "Set store tensor of block " << node->id() << " to register";
      ir_sch_->SetBuffer(cur_block, "local");
    }
  };
  schedule_block_graph_->DFSTopoWalk(SetStorage);
  VLOG(5) << "[After AllocateStorage] func body: "
          << ir_sch_->GetModule().GetExprs().front();
}

void GroupScheduler::UpdateBlockOrder() {
  ir::Expr root_block = ir_sch_->GetRootBlock(ir_sch_->GetAllBlocks()[0]);
  ir::BlockOrderConstructor block_order_constructor;
  blocks_order_with_ctrl_stmt_ = block_order_constructor(&root_block);
}

bool GroupScheduler::IsKeepGraphDependency(Expr schedule_block,
                                           Expr target_loop,
                                           int insert_pos) const {
  // Assuming inserting the schedule_block into the target_loop,
  // obtain the transformed upstream and downstream blocks.
  std::unordered_set<std::string> blocks_above;
  std::unordered_set<std::string> blocks_below;
  bool is_below = false;
  bool find_target_loop = false;
  int pos_count = -1;
  std::map<std::vector<int>, ir::Expr>::const_iterator iter;
  for (iter = blocks_order_with_ctrl_stmt_.begin();
       iter != blocks_order_with_ctrl_stmt_.end();
       ++iter) {
    if (iter->second.get() == schedule_block.get()) {
      continue;
    }
    if (iter->second.get() == target_loop.get()) {
      find_target_loop = true;
    }
    if (find_target_loop) {
      ++pos_count;
    }
    if (pos_count == insert_pos) {
      is_below = true;
    }
    if (iter->second.As<ir::ScheduleBlockRealize>()) {
      std::string block_id = iter->second.As<ir::ScheduleBlockRealize>()
                                 ->schedule_block.As<ir::ScheduleBlock>()
                                 ->name;
      if (is_below) {
        blocks_below.insert(block_id);
      } else {
        blocks_above.insert(block_id);
      }
    }
  }

  // Obtain real upstream and downstream nodes
  std::string src_id = schedule_block.As<ir::ScheduleBlockRealize>()
                           ->schedule_block.As<ir::ScheduleBlock>()
                           ->name;
  ir::ScheduleBlockNode* node = schedule_block_graph_->RetrieveNode(src_id);
  std::unordered_set<std::string> upstream_ids = node->UpstreamNodes();
  std::unordered_set<std::string> downstream_ids = node->DownstreamNodes();

  // Check that the transformed upstream and downstream blocks
  // still meet the relationship between the
  // original upstream and downstream nodes.
  for (const std::string& id : upstream_ids) {
    if (blocks_above.count(id) == 0) {
      VLOG(6) << "[Breaking Graph Level Dependency] ScheduleBlock: " << src_id
              << " cannot be insert into target loop at insert_pos: "
              << insert_pos << " because its upstream block: " << id
              << " will appear downstream.";
      VLOG(6) << "The target loop:\n" << target_loop;
      return false;
    }
  }
  for (const std::string& id : downstream_ids) {
    if (blocks_below.count(id) == 0) {
      VLOG(6) << "[Breaking Graph Level Dependency] ScheduleBlock: " << src_id
              << " cannot be insert into target loop at insert_pos: "
              << insert_pos << " because its downstream block: " << id
              << " will appear upstream.";
      VLOG(6) << "The target loop:\n" << target_loop;
      return false;
    }
  }
  VLOG(6) << "[Meet Graph Level Dependency] ScheduleBlock: " << src_id
          << " can be insert into target loop at insert_pos: " << insert_pos;
  VLOG(6) << "The target loop:\n" << target_loop;
  return true;
}

bool GroupScheduler::MeetConditions(Expr schedule_block,
                                    Expr target_loop,
                                    int insert_pos) const {
  for (const auto& condition_func : feasible_conditions_) {
    if (!(this->*condition_func)(schedule_block, target_loop, insert_pos)) {
      return false;
    }
  }
  return true;
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
