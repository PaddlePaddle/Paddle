// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <unordered_map>
#include <variant>

#include "paddle/cinn/frontend/group_cluster/group_cluster.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/cinn/hlir/framework/compile_error.h"
#include "paddle/cinn/hlir/framework/pir/op_lowering_util.h"
#include "paddle/cinn/hlir/framework/pir/trivial_op_util.h"
#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/cinn/hlir/op/external_api_registry.h"
#include "paddle/cinn/hlir/pe/map_expr_to_ir.h"
#include "paddle/cinn/ir/dim.h"
#include "paddle/cinn/ir/group_schedule/base_group_scheduler.h"
#include "paddle/cinn/ir/group_schedule/st_shape_group_scheduler.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"
#include "paddle/cinn/ir/schedule/ir_schedule_util.h"
#include "paddle/cinn/lang/placeholder.h"
#include "paddle/cinn/optim/schedule_block_dce.h"
#include "paddle/cinn/optim/transform_gpu_forloop.h"
#include "paddle/common/ddim.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_op.h"

namespace cinn {
namespace hlir {
namespace framework {
namespace pir {
namespace trivial_fusion_detail {

struct TrivialOp {
 public:
  explicit TrivialOp(const ir::Expr& origin_func_body);

  TrivialOp(const TrivialOp& trivial_op);

  void _SetFuncBody(ir::Expr new_body);
  ir::Expr* _GetFuncBodyPointer();

  ir::Expr GetFuncBody() const;

 private:
  ir::Expr func_body;
};

struct ReduceOp {
 public:
  explicit ReduceOp(const ir::Expr& origin_func_body);
  ReduceOp(const ReduceOp& reduce_op);

  void _SetFuncBody(ir::Expr new_body);

  ir::Expr GetFuncBody() const;

  ir::Expr* _GetFuncBodyPointer();

 private:
  ir::Expr func_body;
};

using FusibleOp = std::variant<ReduceOp, TrivialOp>;

ir::Expr _GetRootExpr(const FusibleOp& op);

void _SetFuncBody(FusibleOp& op, ir::Expr new_body);  // NOLINT
ir::Expr GetComputeBody(const FusibleOp& op);

ir::Tensor GetOutputTensor(const FusibleOp& op);

std::vector<ir::Var> AppendBound(const std::vector<ir::Var> vars,
                                 const ir::Expr& root);

std::vector<ir::Var> GetOutputIters(const FusibleOp& op);

std::vector<ir::Var> GetReduceIters(const ReduceOp& op);

ir::Expr GetInitExpr(const ReduceOp& op);

ir::Expr* _GetFuncBodyPointer(FusibleOp op);

ir::Expr CopyReduceBody(const FusibleOp& downstream, const ReduceOp& upstream);

ir::Expr CreateReduceExpr(
    const std::vector<ir::Var>& output_iters,
    const std::vector<ir::Var>& reduce_iters,
    const ir::Expr& init_body,    // relay on output_iters
    const ir::Expr& reduce_body,  // relay on output_iters + reduce_iters
    const ir::Tensor& new_write_tensor,
    const ir::Tensor& origin_write_tensor);

ir::Expr CreateTrivialExpr(const std::vector<ir::Var>& output_iters,
                           const ir::Expr& function_body,
                           const ir::Tensor& new_write_tensor);
ir::Expr CreateExprWithNewComputeBody(const FusibleOp& fusible_op,
                                      const ir::Expr& new_compute_body);
struct FusionNode {
  FusibleOp fusible_op;
  ::pir::Operation* expr_related_op;

  std::unordered_map<FusionNode*, ::pir::Value> upstream;
  std::unordered_map<FusionNode*, ::pir::Value> downstream;

  explicit FusionNode(FusibleOp fusible_op);

  static std::string GetTensorCounter();
  void replace_topo_structure_of_fused_nodes(FusionNode* fused_up_node,
                                             FusionNode* fused_down_node);

  bool IsTrivial() const;
};

bool CheckAllLoopRangeEq(ReduceOp reduce_upper, TrivialOp trivial_down);

FusibleOp CreateFusibleOp(ir::Expr compute_body, OpPatternKind op_pattern);

struct FusionGraph {
  explicit FusionGraph(
      const cinn::frontend::group_cluster::PatternNodePtr& pattern_node,
      const std::unordered_map<::pir::Operation*, ir::Expr>& op_expr_map);
  ~FusionGraph();

  std::vector<ir::Expr> DoFusion();

 private:
  FusionNode* FindTrivialFusibleNode();
  void DoTrivialFusion();
  void ReduceLoopTranform();
  void SplitReduceTransform();
  std::vector<ir::Expr> GetExprResults();
  void RemoveNode(FusionNode* node);
  void AppendNode(FusionNode* node);
  FusionNode* FindReduceUpstream(FusionNode* node);

 private:
  FusibleOp TrivialFusion(FusionNode* upstream, FusionNode* downstream);

  template <class DownStreamOp>
  DownStreamOp TrivalxOther_Fusion(TrivialOp upstream,
                                   DownStreamOp downstream) {
    VLOG(4) << "Trivial x OtherFusion begin.";

    const auto& replaced_tensor = GetOutputTensor(upstream);
    VLOG(4) << "upstream is " << upstream.GetFuncBody();
    VLOG(4) << "downstream is " << downstream.GetFuncBody();

    ir::Expr modified_body = ir::ir_utils::IRCopy(downstream.GetFuncBody());
    SequenceMutator(
        ComposeUtils::GetEachTensorLoadExpr(modified_body, replaced_tensor),
        &modified_body,
        [&](const ir::Expr& downstream_load_expr, ir::Expr* downstream_body) {
          ComposeUtils::ReplaceDownstreamLoadExprWithUpstreamComputeBody(
              upstream, downstream_load_expr, downstream_body);
        });

    VLOG(4) << "TTFusion end:\n" << modified_body;
    return DownStreamOp(modified_body);
  }

  std::vector<FusibleOp> ReduceTransform(FusionNode* downstream);
  std::vector<FusibleOp> ReduceTransformRecursive(FusibleOp root_op,
                                                  FusionNode* fusion_tree);
  std::vector<FusibleOp> TransformReduceLoopRange(const ReduceOp& upstream,
                                                  FusibleOp* downstream);
  FusibleOp SinkTrivialLoopAlign(TrivialOp trivial_op, ReduceOp reduce_op);

  template <typename T>
  std::vector<T> FilterWithFakeReduceIter(const std::vector<T>& input) {
    std::vector<T> result;
    for (size_t i = 0; i < input.size(); i++) {
      if (std::find(fake_reduce_iter_idx_.begin(),
                    fake_reduce_iter_idx_.end(),
                    i) == fake_reduce_iter_idx_.end()) {
        result.emplace_back(input.at(i));
      }
    }
    return result;
  }

 private:
  std::unordered_set<FusionNode*> all_fusion_nodes_;
  std::vector<FusibleOp> fusion_results_;
  std::unordered_set<FusionNode*> entrance_nodes_;
  std::unordered_set<FusionNode*> exit_nodes_;

  std::vector<size_t> fake_reduce_iter_idx_;
  // std::unordered_map<::pir::Value, ShardableAxes> shardable_axes_;
};

}  // namespace trivial_fusion_detail

struct FusionGroupInfo {
  std::vector<int64_t> loop_ranges;
  std::vector<int64_t> reduce_axis;
  std::vector<std::string> reduce_var_name;

  std::string DebugPrint() {
    return "GroupInfo\nloop_ranges: " + cinn::utils::Join(loop_ranges, " ") +
           "\nreduce_axis: " + cinn::utils::Join(reduce_axis, " ") +
           "\nreduce_var_name: " + cinn::utils::Join(reduce_var_name, " ");
  }
};

FusionGroupInfo GetFusionGroupInfo(
    const std::vector<ir::Expr>& op_compute_bodies);

std::vector<ir::Expr> OperationFusion(
    const std::vector<::pir::Operation*>& ops,
    const std::vector<ir::Expr>& op_compute_bodies);

}  // namespace pir
}  // namespace framework
}  // namespace hlir
}  // namespace cinn
