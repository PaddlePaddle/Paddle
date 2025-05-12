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

#include "paddle/ap/include/paddle/pass/ap_generic_drr_pass.h"
#include "paddle/ap/include/memory/circlable_ref_list_base.h"

#include "paddle/ap/include/adt/topo_walker.h"
#include "paddle/ap/include/axpr/abstract_list.h"
#include "paddle/ap/include/axpr/anf_expr_util.h"
#include "paddle/ap/include/axpr/atomic.h"
#include "paddle/ap/include/axpr/builtin_frame_util.h"
#include "paddle/ap/include/axpr/builtin_serializable_attr_map_to_axpr_helper.h"
#include "paddle/ap/include/axpr/cps_interpreter.h"
#include "paddle/ap/include/axpr/data_type_util.h"
#include "paddle/ap/include/axpr/lambda_expr_builder.h"
#include "paddle/ap/include/code_gen/arg_source_maker.h"
#include "paddle/ap/include/code_gen/matched_result_pattern_helper.h"
#include "paddle/ap/include/code_gen/value.h"
#include "paddle/ap/include/code_module/module_to_axpr_helper.h"
#include "paddle/ap/include/drr/drr_graph_descriptor.h"
#include "paddle/ap/include/drr/drr_node_descriptor.h"
#include "paddle/ap/include/drr/drr_pass_type_helper.h"
#include "paddle/ap/include/drr/res_ptn_packed_ir_op_declare_data.h"
#include "paddle/ap/include/drr/result_pattern_helper.h"
#include "paddle/ap/include/drr/value.h"
#include "paddle/ap/include/graph/graph_helper.h"
#include "paddle/ap/include/ir_match/graph_matcher.h"
#include "paddle/ap/include/ir_match/ir_match_ctx.h"
#include "paddle/ap/include/ir_match/op_match_ctx_method_class.h"
#include "paddle/ap/include/ir_match/tensor_match_ctx_method_class.h"
#include "paddle/ap/include/paddle/hlir/manual_op.h"
#include "paddle/ap/include/paddle/pass/ap_drr_helper.h"
#include "paddle/ap/include/paddle/pass/ap_kernel_define_helper.h"
#include "paddle/ap/include/paddle/pass/ap_registry_helper.h"
#include "paddle/ap/include/paddle/pass/ir_helper_method_class.h"
#include "paddle/ap/include/paddle/pir/pir_method_class.h"
#include "paddle/ap/include/paddle/pir/pir_node_matched_src_ptn_ctx_helper.h"
#include "paddle/ap/include/paddle/pir/pir_to_anf_expr_helper.h"
#include "paddle/ap/include/paddle/pir/program_method_class.h"
#include "paddle/ap/include/paddle/pir_graph_descriptor.h"
#include "paddle/ap/include/paddle/pir_node.h"
#include "paddle/ap/include/paddle/pir_node_descriptor.h"
#include "paddle/ap/include/reified_drr/reified_drr_pass_dump_helper.h"
#include "paddle/ap/src/paddle/pass/op_factory.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/op_attribute.h"
#include "paddle/cinn/hlir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/src/ir_operation_factory.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace cinn::dialect::ir {

namespace adt = ap::adt;

namespace {

using ap::paddle::PirNode;

using DrrValue = ap::drr::Value;
using DrrNode = ap::drr::Node;

using DrrCtx = ap::drr::DrrCtx;

using DrrNativeIrValue = ap::drr::NativeIrValue<DrrNode>;
using DrrPackedIrValue = ap::drr::PackedIrValue<DrrNode>;
using DrrIrValue = ap::drr::IrValue;

using DrrNativeIrOp = ap::drr::NativeIrOp<DrrNode>;
using DrrNativeIrOpOperand = ap::drr::NativeIrOpOperand<DrrNode>;
using DrrNativeIrOpResult = ap::drr::NativeIrOpResult<DrrNode>;
using DrrPackedIrOp = ap::drr::PackedIrOp<DrrNode>;
using DrrPackedIrOpOperand = ap::drr::PackedIrOpOperand<DrrNode>;
using DrrPackedIrOpResult = ap::drr::PackedIrOpResult<DrrNode>;
using DrrOptPackedIrOp = ap::drr::OptPackedIrOp<DrrNode>;
using DrrOptPackedIrOpOperand = ap::drr::OptPackedIrOpOperand<DrrNode>;
using DrrOptPackedIrOpResult = ap::drr::OptPackedIrOpResult<DrrNode>;

using DrrIrOpImpl = std::variant<DrrNativeIrOp, DrrPackedIrOp>;

using IrMatchCtx = ap::ir_match::IrMatchCtx<PirNode>;

template <typename NodeT>
using NativeORGraph =
    ap::graph::GraphDescriptor<NodeT,
                               ap::drr::topo_kind::NativeOperandAndResult>;

template <typename NodeT>
using DefaultGraph =
    ap::graph::GraphDescriptor<NodeT, ap::drr::topo_kind::Default>;

template <typename NodeT>
using RefAugmentedGraph =
    ap::graph::GraphDescriptor<NodeT, ap::drr::topo_kind::RefAugmented>;

using ap::axpr::AnfExpr;
using CGValue = ap::code_gen::Value;
using CodeGenCtx = ap::code_gen::CodeGenCtx<PirNode>;
using CodeGenResult = ap::code_gen::CodeGenResult<ap::axpr::Value>;
using ap::code_module::CodeModule;

struct DrrIrOp : public DrrIrOpImpl {
  using DrrIrOpImpl::DrrIrOpImpl;
  ADT_DEFINE_VARIANT_METHODS(DrrIrOpImpl);

  const std::string& op_name() const {
    return Match([](const auto& impl) -> const std::string& {
      return impl->op_declare->op_name;
    });
  }
};
using DrrGraphNode = ap::graph::Node<DrrNode>;
using GraphMatchCtx = ap::ir_match::GraphMatchCtx<PirNode>;

using PirNativeIrValue = ap::paddle::NativeIrValue;
using PirNativeIrOpOperand = ap::paddle::NativeIrOpOperand;
using PirNativeIrOpResult = ap::paddle::NativeIrOpResult;

adt::Result<DrrNode> GetApDrrDefaultAnchor(const DrrCtx& drr_ctx) {
  ADT_LET_CONST_REF(src_ptn_ctx, drr_ctx->GetSourcePatternCtx());
  auto ptn_node_area = src_ptn_ctx->node_arena;
  ap::graph::GraphDescriptor<DrrGraphNode, ap::drr::topo_kind::Default>
      source_pattern_graph{};
  ADT_CHECK(ptn_node_area->nodes().size() > 0);
  ap::graph::GraphHelper<DrrGraphNode, ap::drr::topo_kind::Default>
      graph_helper(source_pattern_graph);
  const auto& start_ptn_node = ptn_node_area->nodes().at(0).node();
  ADT_LET_CONST_REF(anchor_node, graph_helper.FindAnchor(start_ptn_node));
  ADT_LET_CONST_REF(default_anchor, anchor_node.Get());
  return default_anchor;
}

adt::Result<std::optional<DrrNativeIrOp>> GetApDrrNativeIrOpAnchor(
    const DrrCtx& drr_ctx) {
  ADT_LET_CONST_REF(src_ptn_ctx, drr_ctx->GetSourcePatternCtx());
  auto ptn_node_area = src_ptn_ctx->node_arena;
  ap::graph::GraphDescriptor<DrrGraphNode, ap::drr::topo_kind::Default>
      source_pattern_graph{};
  ADT_CHECK(ptn_node_area->nodes().size() > 0);
  ap::graph::GraphHelper<DrrGraphNode, ap::drr::topo_kind::Default>
      graph_helper(source_pattern_graph);
  const auto& start_ptn_node = ptn_node_area->nodes().at(0).node();
  auto IsNativeOpWithOutputs = [&](const auto& node) -> adt::Result<bool> {
    ADT_LET_CONST_REF(drr_node, node.Get());
    ADT_LET_CONST_REF(downstreams, node.DownstreamNodes());
    return drr_node.template Has<DrrNativeIrOp>() && downstreams.size() > 0;
  };
  const auto& Filter = IsNativeOpWithOutputs;
  ADT_LET_CONST_REF(anchor_node,
                    graph_helper.FilterAnchor(start_ptn_node, Filter));
  if (!anchor_node.has_value()) {
    return std::nullopt;
  }
  ADT_LET_CONST_REF(anchor, anchor_node.value().Get());
  ADT_LET_CONST_REF(native_ir_op, anchor.template TryGet<DrrNativeIrOp>());
  return native_ir_op;
}

adt::Result<std::vector<DrrIrValue>> GetResPtnInputs(const DrrCtx& drr_ctx) {
  std::vector<DrrIrValue> ret;
  ADT_LET_CONST_REF(res_ptn_ctx, drr_ctx->GetResultPatternCtx());
  const auto& nodes = res_ptn_ctx->node_arena->nodes();
  for (const auto& drr_node : nodes) {
    ADT_LET_CONST_REF(upstreams, drr_node.node().UpstreamNodes());
    if (upstreams.size() == 0) {
      const auto& opt_drr_ir_value = DrrIrValue::OptCastFrom(drr_node);
      if (opt_drr_ir_value.has_value()) {
        ret.push_back(opt_drr_ir_value.value());
      }
    }
  }
  return ret;
}

adt::Result<std::vector<DrrIrValue>> GetResPtnOutputs(const DrrCtx& drr_ctx) {
  std::vector<DrrIrValue> ret;
  ADT_LET_CONST_REF(res_ptn_ctx, drr_ctx->GetResultPatternCtx());
  const auto& nodes = res_ptn_ctx->node_arena->nodes();
  for (const auto& drr_node : nodes) {
    ADT_LET_CONST_REF(downstreams, drr_node.node().DownstreamNodes());
    if (downstreams.size() == 0) {
      const auto& opt_drr_ir_value = DrrIrValue::OptCastFrom(drr_node);
      if (opt_drr_ir_value.has_value()) {
        ret.push_back(opt_drr_ir_value.value());
      }
    }
  }
  return ret;
}

class DrrCtxProvider {
 public:
  DrrCtxProvider() {}

  virtual adt::Result<adt::List<DrrCtx>> GetDrrCtxList() = 0;

  virtual adt::Result<adt::Ok> PostProcess(
      adt::Result<std::optional<GraphMatchCtx>> (*Match)(const DrrCtx&,
                                                         pir::Operation* op),
      const DrrCtx& drr_ctx,
      pir::Operation* op,
      const GraphMatchCtx& match_ctx,
      const std::function<adt::Result<CodeGenResult>(const std::string&)>&
          CodeGenResult4FusedOpName) = 0;
};

class NaiveDrrCtxProvider : public DrrCtxProvider {
  DrrCtx drr_ctx_;

 public:
  explicit NaiveDrrCtxProvider(const DrrCtx& drr_ctx) : drr_ctx_(drr_ctx) {}

  adt::Result<adt::List<DrrCtx>> GetDrrCtxList() override {
    return adt::List<DrrCtx>{drr_ctx_};
  }

  adt::Result<adt::Ok> PostProcess(
      adt::Result<std::optional<GraphMatchCtx>> (*Match)(const DrrCtx&,
                                                         pir::Operation* op),
      const DrrCtx& drr_ctx,
      pir::Operation* op,
      const GraphMatchCtx& match_ctx,
      const std::function<adt::Result<CodeGenResult>(const std::string&)>&
          CodeGenResult4FusedOpName) override {
    // Do Nothing.
    return adt::Ok{};
  }
};

struct ApGenericDrrPatternCtx {
  std::shared_ptr<DrrCtxProvider> drr_ctx_provider_;
  DrrCtx drr_ctx;
  std::vector<DrrIrValue> res_ptn_outputs;
  DrrNode default_anchor;
  std::optional<DrrNativeIrOp> native_op_anchor;
  std::string anchor_op_name;
  std::optional<int64_t> steps_limit;

  const std::shared_ptr<DrrCtxProvider>& drr_ctx_provider() const {
    return drr_ctx_provider_;
  }

  static adt::Result<ApGenericDrrPatternCtx> MakeFromDrrCtx(
      const DrrCtx& drr_ctx,
      std::optional<int64_t> steps_limit,
      const std::shared_ptr<DrrCtxProvider>& drr_ctx_provider) {
    ADT_LET_CONST_REF(res_ptn_outputs, GetResPtnOutputs(drr_ctx));
    ADT_LET_CONST_REF(default_anchor, GetApDrrDefaultAnchor(drr_ctx));
    std::optional<DrrNativeIrOp> opt_native_op_anchor;
    if (ap::drr::DrrPassTypeHelper{}.SupportOptionalPackedOp(
            drr_ctx->drr_pass_type)) {
      ADT_LET_CONST_REF(opt_native_ir_op_anchor,
                        GetApDrrNativeIrOpAnchor(drr_ctx));
      opt_native_op_anchor = opt_native_ir_op_anchor;
    } else {
      opt_native_op_anchor = std::nullopt;
    }
    ADT_LET_CONST_REF(anchor_op_name,
                      GetAnchorOpName(opt_native_op_anchor, default_anchor));
    return ApGenericDrrPatternCtx{drr_ctx_provider,
                                  drr_ctx,
                                  res_ptn_outputs,
                                  default_anchor,
                                  opt_native_op_anchor,
                                  anchor_op_name,
                                  steps_limit};
  }

  static adt::Result<std::string> GetAnchorOpName(
      const std::optional<DrrNativeIrOp>& native_op_anchor,
      const DrrNode& default_anchor) {
    if (native_op_anchor.has_value()) {
      return native_op_anchor.value()->op_declare->op_name;
    }
    return default_anchor.Match(
        [&](const DrrNativeIrOp& ir_op) -> adt::Result<std::string> {
          return ir_op->op_declare->op_name;
        },
        [&](const DrrPackedIrOp& ir_op) -> adt::Result<std::string> {
          return PirNode::GetOpNameFromDrrPackedOpName(
              ir_op->op_declare->op_name);
        },
        [&](const DrrOptPackedIrOp& ir_op) -> adt::Result<std::string> {
          return PirNode::GetOpNameFromDrrPackedOpName(
              ir_op->op_declare->op_name);
        },
        [&](const auto&) -> adt::Result<std::string> {
          return adt::errors::TypeError{
              "default_anchor drr node should be a op node but value node "
              "found."};
        });
  }
};

struct ApRewriter {
  ApGenericDrrPatternCtx ctx_;
  adt::Result<std::optional<GraphMatchCtx>> (*Match_)(const DrrCtx&,
                                                      pir::Operation* op);
  mutable ApDrrHelper ap_drr_helper_;

  ApRewriter(const ApGenericDrrPatternCtx& ctx,
             adt::Result<std::optional<GraphMatchCtx>> (*Match)(
                 const DrrCtx&, pir::Operation* op))
      : ctx_(ctx),
        Match_(Match),
        ap_drr_helper_(ctx_.drr_ctx->circlable_ref_list) {}

  adt::Result<bool> Rewrite(const GraphMatchCtx& match_ctx,
                            pir::Operation* op,
                            pir::PatternRewriter* rewriter) const {
    ADT_CHECK(ctx_.drr_ctx->pass_name.has_value());
    VLOG(0) << "drr: " << ctx_.drr_ctx->pass_name.value() << " matched.";
    return RewriteByResultPattern(match_ctx, op, rewriter);
  }

 private:
  adt::Result<bool> RewriteByResultPattern(
      const GraphMatchCtx& match_ctx,
      pir::Operation* op,
      pir::PatternRewriter* rewriter) const {
    ADT_LET_CONST_REF(rewritten,
                      TryRewriteByResultPattern(match_ctx, op, rewriter));
    return rewritten;
  }

  using IrValue2UseIterators =
      std::unordered_map<pir::Value, std::list<pir::Value::UseIterator>>;

  struct RewriteCtx {
    std::unordered_map<pir::Operation*, int64_t> matched_op2order_value;
    IrValue2UseIterators output2original_uses;
    std::unordered_map<std::string, pir::Value> name2native_value;
    std::unordered_map<std::string, std::vector<pir::Value>> name2packed_values;

    adt::Result<int64_t> GetMatchedOpOrderValue(pir::Operation* op) const {
      const auto iter = this->matched_op2order_value.find(op);
      if (iter == this->matched_op2order_value.end()) {
        return adt::errors::IndexError{
            "RewriteCtx::GetMatchedOpOrderValue failed."};
      }
      return iter->second;
    }

    adt::Result<pir::Value> GetNativeIrValue(
        const std::string& ir_value_name) const {
      const auto iter = this->name2native_value.find(ir_value_name);
      if (iter == this->name2native_value.end()) {
        return adt::errors::IndexError{
            "RewriteCtx::GetNativeIrValue() failed. key '" + ir_value_name +
            "' not found."};
      }
      return iter->second;
    }

    adt::Result<const std::vector<pir::Value>*> GetPackedIrValues(
        const std::string& ir_value_name) const {
      const auto iter = this->name2packed_values.find(ir_value_name);
      if (iter == this->name2packed_values.end()) {
        return adt::errors::IndexError{
            "RewriteCtx::GetPackedIrValues() failed. key '" + ir_value_name +
            "' not found"};
      }
      return &iter->second;
    }
  };

  adt::Result<std::unordered_set<pir::Operation*>> GetMatchedOps(
      const GraphMatchCtx& match_ctx) const {
    using DefaultDrrGraph =
        ap::graph::GraphDescriptor<DrrGraphNode, ap::drr::topo_kind::Default>;
    DefaultDrrGraph default_drr_graph{};
    ADT_LET_CONST_REF(src_ptn_ctx, ctx_.drr_ctx->GetSourcePatternCtx());
    const auto& nodes = src_ptn_ctx->node_arena->nodes();
    std::unordered_set<pir::Operation*> ops;
    for (const auto& drr_node : nodes) {
      ADT_LET_CONST_REF(is_op_node,
                        default_drr_graph.IsOpNode(drr_node.node()));
      if (!is_op_node) {
        continue;
      }
      ADT_LET_CONST_REF(pir_node,
                        match_ctx->GetSoleBigGraphNode(drr_node.node()));
      const auto& opt_op = CastToPirOp(pir_node);
      if (opt_op.has_value()) {
        ADT_CHECK(ops.emplace(opt_op.value()).second);
      }
    }
    return ops;
  }

  std::optional<pir::Operation*> CastToPirOp(const PirNode& pir_node) const {
    return pir_node.Match(
        [](const ap::paddle::NativeIrOp& ir_op)
            -> std::optional<pir::Operation*> { return ir_op.op; },
        [&](const ap::paddle::PackedIrOp& ir_op)
            -> std::optional<pir::Operation*> {
          return static_cast<pir::Operation*>(ir_op.fusion_op);
        },
        [&](const auto&) -> std::optional<pir::Operation*> {
          return std::nullopt;
        });
  }

  adt::Result<adt::Ok> InitRewriteCtx(
      RewriteCtx* rewrite_ctx, const GraphMatchCtx& graph_match_ctx) const {
    ADT_LET_CONST_REF(matched_op2order_value_map,
                      MakeMatchedOp2OrderValue(graph_match_ctx));
    rewrite_ctx->matched_op2order_value = matched_op2order_value_map;
    auto* map = &rewrite_ctx->output2original_uses;
    ADT_RETURN_IF_ERR(InitResPtnOutput2UseIterators(map, graph_match_ctx));
    return adt::Ok{};
  }

  adt::Result<adt::Ok> InitResPtnOutput2UseIterators(
      IrValue2UseIterators* map, const GraphMatchCtx& graph_match_ctx) const {
    auto UpdateValue2Use = [&](pir::Value output) -> adt::Result<adt::Ok> {
      auto* lst = &(*map)[output];
      for (auto iter = output.use_begin(); iter != output.use_end(); ++iter) {
        lst->emplace_back(iter);
      }
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(
        VisitResPtnOutputPirValue(graph_match_ctx, UpdateValue2Use));
    return adt::Ok{};
  }

  adt::Result<std::unordered_map<pir::Operation*, int64_t>>
  MakeMatchedOp2OrderValue(const GraphMatchCtx& match_ctx) const {
    ADT_LET_CONST_REF(ops, GetMatchedOps(match_ctx));
    return MakeMatchedOp2OrderValue(ops);
  }

  adt::Result<std::unordered_map<pir::Operation*, int64_t>>
  MakeMatchedOp2OrderValue(
      const std::unordered_set<pir::Operation*>& ops) const {
    std::unordered_map<pir::Operation*, int64_t> ret;
    if (ops.empty()) {
      return ret;
    }
    pir::Operation* start = *ops.begin();
    auto* block = start->GetParent();
    pir::Block::Iterator left_iter = *start;
    pir::Block::Iterator right_iter = *start;
    for (int i = 0; ret.size() < ops.size() && i < block->size(); ++i) {
      if (ops.count(&*left_iter) > 0) {
        ret[&*left_iter] = -i;
      }
      if (ops.count(&*right_iter) > 0) {
        ret[&*right_iter] = i;
      }
      if (&*left_iter != &block->front()) {
        --left_iter;
      }
      if (&*right_iter != &block->back()) {
        ++right_iter;
      }
    }
    ADT_CHECK(ret.size() == ops.size());
    return ret;
  }

  using CodeGenResultCollectT = std::function<adt::Result<adt::Ok>(
      const std::string& fused_op_name, const CodeGenResult&)>;

  adt::Result<bool> TryRewriteByResultPattern(
      const GraphMatchCtx& match_ctx,
      pir::Operation* op,
      pir::PatternRewriter* rewriter) const {
    ADT_RETURN_IF_ERR(WithPostProcessGuard(
        match_ctx,
        op,
        [&](const auto& CodeGenResultCollect) -> adt::Result<adt::Ok> {
          RewriteCtx rewrite_ctx;
          ADT_RETURN_IF_ERR(InitRewriteCtx(&rewrite_ctx, match_ctx));
          auto Build = [&](const auto& res_ptn_op) -> adt::Result<adt::Ok> {
            return BuildNewOp(rewriter,
                              res_ptn_op,
                              &rewrite_ctx,
                              match_ctx,
                              CodeGenResultCollect);
          };
          ADT_RETURN_IF_ERR(VisitEachResPtnOp(Build));
          ADT_RETURN_IF_ERR(
              ReplaceOutputResPtnTensor(match_ctx, rewrite_ctx, rewriter));
          return adt::Ok{};
        }));
    return true;
  }

  template <typename DoWithCollectorT>
  adt::Result<adt::Ok> WithPostProcessGuard(
      const GraphMatchCtx& match_ctx,
      pir::Operation* op,
      const DoWithCollectorT& DoWithCollector) const {
    std::map<std::string, CodeGenResult> fused_op_name2code_gen_result;
    auto CodeGenResultCollect =
        [&](const std::string& fused_op_name,
            const CodeGenResult& code_gen_result) -> adt::Result<adt::Ok> {
      ADT_CHECK(
          fused_op_name2code_gen_result.emplace(fused_op_name, code_gen_result)
              .second);
      return adt::Ok{};
    };

    ADT_RETURN_IF_ERR(DoWithCollector(CodeGenResultCollect));

    using RetT = adt::Result<CodeGenResult>;
    auto CodeGenResult4FusedOpName =
        [&](const std::string& fused_op_name) -> RetT {
      const auto& iter = fused_op_name2code_gen_result.find(fused_op_name);
      ADT_CHECK(iter != fused_op_name2code_gen_result.end());
      return iter->second;
    };
    ADT_RETURN_IF_ERR(ctx_.drr_ctx_provider()->PostProcess(
        Match_, ctx_.drr_ctx, op, match_ctx, CodeGenResult4FusedOpName));
    return adt::Ok{};
  }

  adt::Result<adt::Ok> ReplaceOutputResPtnTensor(
      const GraphMatchCtx& match_ctx,
      const RewriteCtx& rewrite_ctx,
      pir::PatternRewriter* rewriter) const {
    auto Replace = [&](pir::Value from, pir::Value to) -> adt::Result<adt::Ok> {
      // Reason for no use of `rewriter->ReplaceAllUsesWith(from, to)`:
      // AP drr mechanism support result pattern like:
      //   o.foo_op(
      //     [o.bar_value],
      //     [o.bar_value]
      //   )
      // It will insert `foo_op` between pir::Value named `bar_value` and its
      // consumer ops except the newly inserted `foo_op`.
      auto iter = rewrite_ctx.output2original_uses.find(from);
      ADT_CHECK(iter != rewrite_ctx.output2original_uses.end());
      for (auto use_iter : iter->second) {
        use_iter->set_source(to);
      }
      return adt::Ok{};
    };
    return VisitOutputPirValueReplacementPair(match_ctx, rewrite_ctx, Replace);
  }

  template <typename YieldT>
  adt::Result<adt::Ok> VisitResPtnOutputPirValue(const GraphMatchCtx& match_ctx,
                                                 const YieldT& Yield) const {
    for (const auto& res_ptn_drr_ir_value : ctx_.res_ptn_outputs) {
      const auto& opt_drr_ir_value =
          SrcPtnIrValue4ResPtnIrValue(res_ptn_drr_ir_value);
      ADT_CHECK(opt_drr_ir_value.has_value());
      const auto& drr_ir_value = opt_drr_ir_value.value();
      const auto& ret = drr_ir_value.Match(
          [&](const DrrNativeIrValue& native_ir_value) -> adt::Result<adt::Ok> {
            ADT_LET_CONST_REF(
                pir_node,
                match_ctx->GetSoleBigGraphNode(native_ir_value->node));
            ADT_LET_CONST_REF(
                pir_value,
                pir_node.template TryGet<ap::paddle::NativeIrValue>());
            return Yield(pir_value.value);
          },
          [&](const DrrPackedIrValue& packed_ir_value) -> adt::Result<adt::Ok> {
            ADT_LET_CONST_REF(from_nodes,
                              match_ctx->GetPackedBigGraphIrValueNodes(
                                  packed_ir_value->node));
            for (int i = 0; i < from_nodes->size(); ++i) {
              const auto& from_node = from_nodes->at(i);
              ADT_LET_CONST_REF(
                  pir_value,
                  from_node.template TryGet<ap::paddle::NativeIrValue>());
              ADT_RETURN_IF_ERR(Yield(pir_value.value));
            }
            return adt::Ok{};
          });
      ADT_RETURN_IF_ERR(ret);
    }
    return adt::Ok{};
  }

  template <typename DoEachPairT>
  adt::Result<adt::Ok> VisitOutputPirValueReplacementPair(
      const GraphMatchCtx& match_ctx,
      const RewriteCtx& rewrite_ctx,
      const DoEachPairT& DoEachPair) const {
    for (const auto& res_ptn_drr_ir_value : ctx_.res_ptn_outputs) {
      const auto& opt_drr_ir_value =
          SrcPtnIrValue4ResPtnIrValue(res_ptn_drr_ir_value);
      ADT_CHECK(opt_drr_ir_value.has_value());
      const auto& drr_ir_value = opt_drr_ir_value.value();
      const auto& ret = drr_ir_value.Match(
          [&](const DrrNativeIrValue& native_ir_value) -> adt::Result<adt::Ok> {
            ADT_LET_CONST_REF(
                pir_node,
                match_ctx->GetSoleBigGraphNode(native_ir_value->node));
            ADT_LET_CONST_REF(
                pir_value,
                pir_node.template TryGet<ap::paddle::NativeIrValue>());
            pir::Value from = pir_value.value;
            ADT_LET_CONST_REF(
                to, rewrite_ctx.GetNativeIrValue(native_ir_value->name));
            return DoEachPair(from, to);
          },
          [&](const DrrPackedIrValue& packed_ir_value) -> adt::Result<adt::Ok> {
            ADT_LET_CONST_REF(from_nodes,
                              match_ctx->GetPackedBigGraphIrValueNodes(
                                  packed_ir_value->node));
            ADT_LET_CONST_REF(
                to_values_ptr,
                rewrite_ctx.GetPackedIrValues(packed_ir_value->name));
            ADT_CHECK(from_nodes->size() == to_values_ptr->size())
                << adt::errors::ValueError{
                       "from_nodes->size(): " +
                       std::to_string(from_nodes->size()) +
                       ", to_values_ptr->size(): " +
                       std::to_string(to_values_ptr->size()) + "."};
            for (int i = 0; i < from_nodes->size(); ++i) {
              const auto& from_node = from_nodes->at(i);
              ADT_LET_CONST_REF(
                  pir_value,
                  from_node.template TryGet<ap::paddle::NativeIrValue>());
              pir::Value from = pir_value.value;
              pir::Value to = to_values_ptr->at(i);
              ADT_RETURN_IF_ERR(DoEachPair(from, to));
            }
            return adt::Ok{};
          });
      ADT_RETURN_IF_ERR(ret);
    }
    return adt::Ok{};
  }

  template <typename YieldT>
  adt::Result<adt::Ok> VisitEachResPtnOp(const YieldT& Yield) const {
    auto DoEachResPtnOp =
        [&](const auto& res_ptn_graph_node) -> adt::Result<adt::Ok> {
      ADT_LET_CONST_REF(res_ptn_node, res_ptn_graph_node.Get());
      const auto& opt_res_ptn_op = ConvertToResPtnOp(res_ptn_node);
      if (opt_res_ptn_op.has_value()) {
        ADT_RETURN_IF_ERR(Yield(opt_res_ptn_op.value()));
      }
      return adt::Ok{};
    };
    return VisitEachResPtnGraphNode(DoEachResPtnOp);
  }

  template <typename YieldT>
  adt::Result<adt::Ok> VisitEachResPtnGraphNode(const YieldT& Yield) const {
    ADT_LET_CONST_REF(res_ptn_ctx, ctx_.drr_ctx->GetResultPatternCtx());
    std::list<DrrGraphNode> sources;
    for (const auto& drr_node : res_ptn_ctx->node_arena->nodes()) {
      const auto& drr_graph_node = drr_node.node();
      ADT_LET_CONST_REF(upstreams, drr_graph_node.UpstreamNodes());
      if (upstreams.size() == 0) {
        sources.push_back(drr_graph_node);
      }
    }
    using Ok = adt::Result<adt::Ok>;
    ap::drr::DefaultDrrGraphDescriptor graph{};
    auto VisitPrev = [&](const DrrGraphNode& node, const auto& Yield) -> Ok {
      return graph.VisitUpstreamNodes(node, Yield);
    };
    auto VisitNext = [&](const DrrGraphNode& node, const auto& Yield) -> Ok {
      return graph.VisitDownstreamNodes(node, Yield);
    };
    ap::adt::TopoWalker<DrrGraphNode> walker{VisitPrev, VisitNext};
    ADT_RETURN_IF_ERR(walker(sources.begin(), sources.end(), Yield));
    return adt::Ok{};
  }

  std::optional<DrrIrOp> ConvertToResPtnOp(const DrrNode& drr_node) const {
    return drr_node.Match(
        [&](const DrrNativeIrOp& ir_op) -> std::optional<DrrIrOp> {
          return DrrIrOp{ir_op};
        },
        [&](const DrrPackedIrOp& ir_op) -> std::optional<DrrIrOp> {
          return DrrIrOp{ir_op};
        },
        [&](const auto&) -> std::optional<DrrIrOp> { return std::nullopt; });
  }

  adt::Result<adt::Ok> BuildNewOp(
      pir::PatternRewriter* rewriter,
      const DrrIrOp& res_ptn_op,
      RewriteCtx* rewrite_ctx,
      const GraphMatchCtx& match_ctx,
      const CodeGenResultCollectT& CodeGenResultCollect) const {
    return res_ptn_op.Match(
        [&](const DrrNativeIrOp& ir_op) -> adt::Result<adt::Ok> {
          return BuildNativeOp(
              rewriter, ir_op, rewrite_ctx, match_ctx, CodeGenResultCollect);
        },
        [&](const DrrPackedIrOp& ir_op) -> adt::Result<adt::Ok> {
          return BuildPackedOp(
              rewriter, ir_op, rewrite_ctx, match_ctx, CodeGenResultCollect);
        });
  }

  adt::Result<adt::Ok> BuildNativeOp(
      pir::PatternRewriter* rewriter,
      const DrrNativeIrOp& res_ptn_ir_op,
      RewriteCtx* rewrite_ctx,
      const GraphMatchCtx& match_ctx,
      const CodeGenResultCollectT& CodeGenResultCollect) const {
    ADT_RETURN_IF_ERR(
        InsertInputPirValueToReplaceCtx(res_ptn_ir_op, rewrite_ctx, match_ctx));
    ADT_LET_CONST_REF(input_values,
                      GetNativeOpInputValues(res_ptn_ir_op, *rewrite_ctx));
    ADT_RETURN_IF_ERR(
        TrySetInsertPointer(rewriter, *rewrite_ctx, input_values, match_ctx));
    ADT_LET_CONST_REF(attributes,
                      GetResPtnOpAttributes(res_ptn_ir_op, match_ctx));
    ADT_LET_CONST_REF(
        output_values,
        ConstructNativeOp(rewriter, res_ptn_ir_op, input_values, attributes));
    ADT_RETURN_IF_ERR(UpdateConstructedOpOutputsInReplaceCtx(
        match_ctx, output_values, res_ptn_ir_op, rewrite_ctx));
    return adt::Ok{};
  }

  adt::Result<pir::AttributeMap> GetResPtnOpAttributes(
      const DrrNativeIrOp& res_ptn_ir_op,
      const GraphMatchCtx& match_ctx) const {
    ADT_CHECK(ctx_.drr_ctx->source_pattern_ctx.has_value());
    IrMatchCtx ir_match_ctx{ctx_.drr_ctx->source_pattern_ctx.value(),
                            match_ctx};
    const auto& args = GetResPtnAttrGetterArgs(ir_match_ctx);
    pir::AttributeMap attrs;
    auto* drr_interpreter = ap_drr_helper_.mut_drr_interpreter();
    using Ok = adt::Result<adt::Ok>;
    auto CollectAttr = [&](const auto& attr_name, const auto& getter) -> Ok {
      ADT_LET_CONST_REF(attr_val, drr_interpreter->Interpret(getter, args));
      ADT_CHECK(ctx_.drr_ctx->pass_name.has_value());
      ADT_LET_CONST_REF(attr, attr_val.template CastTo<pir::Attribute>());
      ADT_CHECK(attrs.emplace(attr_name, attr).second);
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(VisitResPtnOpAttr(res_ptn_ir_op, CollectAttr));
    return attrs;
  }

  std::vector<ap::axpr::Value> GetResPtnAttrGetterArgs(
      const IrMatchCtx& ir_match_ctx) const {
    ap::ir_match::OpMatchCtx<PirNode> op_match_ctx{ir_match_ctx.shared_ptr()};
    ap::ir_match::TensorMatchCtx<PirNode> tensor_match_ctx{
        ir_match_ctx.shared_ptr()};
    return std::vector<ap::axpr::Value>{
        ap::ir_match::GetOpMatchCtxClass<ap::axpr::Value, PirNode>().New(
            op_match_ctx),
        ap::ir_match::GetTensorMatchCtxClass<ap::axpr::Value, PirNode>().New(
            tensor_match_ctx),
    };
  }

  template <typename YieldT>
  adt::Result<adt::Ok> VisitResPtnOpAttr(const DrrNativeIrOp& res_ptn_ir_op,
                                         const YieldT& Yield) const {
    const auto& attr_map = res_ptn_ir_op->op_declare->attr_map;
    for (const auto& [attr_name, getter] : attr_map->storage) {
      ADT_RETURN_IF_ERR(Yield(attr_name, getter));
    }
    return adt::Ok{};
  }

  adt::Result<std::vector<pir::Value>> ConstructNativeOp(
      pir::PatternRewriter* rewriter,
      const DrrNativeIrOp& res_ptn_ir_op,
      const std::vector<pir::Value>& inputs,
      const pir::AttributeMap& attrs) const {
    {
      ADT_LET_CONST_REF(
          opt_op,
          ap::paddle::CreateOperation(
              rewriter, res_ptn_ir_op->op_declare->op_name, inputs, attrs));
      if (opt_op.has_value()) {
        return opt_op.value()->results();
      }
    }
    try {
      pir::Operation* op =
          paddle::drr::OperationFactory::Instance().CreateOperation(
              res_ptn_ir_op->op_declare->op_name, inputs, attrs, *rewriter);
      return op->results();
    } catch (const std::exception& e) {
      return adt::errors::ValueError{
          std::string() +
          "OperationFactory::Instance().CreateOperation() failed. op_name: " +
          res_ptn_ir_op->op_declare->op_name + ". what(): " + e.what()};
    }
  }

  adt::Result<adt::Ok> BuildPackedOp(
      pir::PatternRewriter* rewriter,
      const DrrPackedIrOp& res_ptn_ir_op,
      RewriteCtx* rewrite_ctx,
      const GraphMatchCtx& match_ctx,
      const CodeGenResultCollectT& CodeGenResultCollect) const {
    ADT_CHECK(res_ptn_ir_op->op_declare->op_name, "ap_pattern_fusion_op");
    ADT_RETURN_IF_ERR(
        InsertInputPirValueToReplaceCtx(res_ptn_ir_op, rewrite_ctx, match_ctx));
    ADT_LET_CONST_REF(input_values,
                      GetPackedOpInputValues(res_ptn_ir_op, *rewrite_ctx));
    ADT_RETURN_IF_ERR(
        TrySetInsertPointer(rewriter, *rewrite_ctx, input_values, match_ctx));
    ADT_LET_CONST_REF(combined_value, InsertCombinedOp(rewriter, input_values));
    ADT_LET_CONST_REF(code_gen_result, CodeGen(res_ptn_ir_op, match_ctx));
    ADT_RETURN_IF_ERR(
        CodeGenResultCollect(res_ptn_ir_op->name, code_gen_result));
    ADT_LET_CONST_REF(
        code_module_anf_expr,
        ConvertApKernelModuleToAnfExpr(code_gen_result->code_module));
    const auto& code_gen_lambda_str = code_module_anf_expr.DumpToJsonString();
    const auto& kernel_dispatch_func = code_gen_result->kernel_dispatch_func;
    const auto& kernel_dispatch_const_data =
        code_gen_result->kernel_dispatch_const_data;
    ADT_LET_CONST_REF(infer_meta_lambda_str,
                      GetInferMetaLambdaStr(res_ptn_ir_op, match_ctx));
    ADT_LET_CONST_REF(kernel_dispatch_lambda_str,
                      GetKernelDispatchLambdaStr(kernel_dispatch_func));
    ADT_LET_CONST_REF(
        kernel_dispatch_const_data_lambda_str,
        GetKernelDispatchConstDataLambdaStr(
            res_ptn_ir_op, match_ctx, kernel_dispatch_const_data));
    ADT_LET_CONST_REF(num_outputs,
                      GetApKernelNumOutputs(res_ptn_ir_op, match_ctx));
    ADT_LET_CONST_REF(
        ap_pattern_fusion_combined_out,
        MakeApPatternFusionOp(rewriter,
                              combined_value,
                              num_outputs,
                              code_gen_lambda_str,
                              infer_meta_lambda_str,
                              kernel_dispatch_lambda_str,
                              kernel_dispatch_const_data_lambda_str));
    ADT_LET_CONST_REF(
        output_values,
        GetPackedOpOutputValues(rewriter, ap_pattern_fusion_combined_out));
    ADT_RETURN_IF_ERR(UpdateConstructedOpOutputsInReplaceCtx(
        match_ctx, output_values, res_ptn_ir_op, rewrite_ctx));
    return adt::Ok{};
  }

  struct InputDimIndex {
    int input_idx;
    int tensor_axis;
  };

  struct OpInferMetaCtx {
    std::unordered_map<symbol::DimExpr, InputDimIndex> dim_expr2in_dim_index;
    mutable std::unordered_map<symbol::DimExpr, AnfExpr> dim_expr2anf_expr;
  };

  struct TensorMeta {
    std::vector<symbol::DimExpr> shape;
    pir::Type dtype;
  };

  adt::Result<std::string> GetInferMetaLambdaStr(
      const DrrPackedIrOp& res_ptn_ir_op,
      const GraphMatchCtx& match_ctx) const {
    ADT_LET_CONST_REF(infer_meta_ctx,
                      GetOpInferMetaCtx(res_ptn_ir_op, match_ctx));
    ADT_LET_CONST_REF(outputs, GetOpOutputPirValues(res_ptn_ir_op, match_ctx));
    auto ConstructLambdaBody =
        [&](ap::axpr::LetContext& ctx) -> adt::Result<AnfExpr> {
      for (int i = 0; i < outputs.size(); ++i) {
        const auto& output = outputs.at(i);
        auto& output_meta_var = ctx.Var("outputs").At(i);
        ADT_LET_CONST_REF(dim_exprs_ptr, GetShapeDimExprsPtrByValue(output));
        ADT_LET_CONST_REF(ddim_val,
                          ConstructDDims(&ctx, infer_meta_ctx, *dim_exprs_ptr));
        output_meta_var.SetAttr("dims", ddim_val);
        ADT_LET_CONST_REF(dtype, GetPirDataType(output));
        ADT_LET_CONST_REF(dtype_val,
                          ConstructDtype(&ctx, infer_meta_ctx, dtype));
        output_meta_var.SetAttr("dtype", dtype_val);
      }
      return ctx.None();
    };
    ap::axpr::LambdaExprBuilder lmbd;
    ADT_LET_CONST_REF(
        anf_expr, lmbd.TryLambda({"inputs", "outputs"}, ConstructLambdaBody));
    return anf_expr.DumpToJsonString();
  }

  adt::Result<pir::Type> GetPirDataType(pir::Value value) const {
    if (!value.type().isa<pir::DenseTensorType>()) {
      return adt::errors::NotImplementedError{
          "pir value must be of DenseTensorType"};
    }
    const auto dense_tensor_type =
        value.type().dyn_cast<pir::DenseTensorType>();
    return dense_tensor_type.dtype();
  }

  adt::Result<std::vector<pir::Value>> GetOpOutputPirValues(
      const DrrPackedIrOp& res_ptn_ir_op,
      const GraphMatchCtx& match_ctx) const {
    return GetMatchedPirOutputsOfRestPtnPackedIrOp(res_ptn_ir_op, match_ctx);
  }

  adt::Result<OpInferMetaCtx> GetOpInferMetaCtx(
      const DrrPackedIrOp& res_ptn_ir_op,
      const GraphMatchCtx& match_ctx) const {
    ADT_LET_CONST_REF(
        inputs,
        GetMatchedPirInputsOfRestPtnPackedIrOp(res_ptn_ir_op, match_ctx));
    OpInferMetaCtx infer_meta_ctx{};
    auto* map = &infer_meta_ctx.dim_expr2in_dim_index;
    for (int in_idx = 0; in_idx < inputs.size(); ++in_idx) {
      pir::Value input = inputs.at(in_idx);
      ADT_LET_CONST_REF(dim_exprs_ptr, GetShapeDimExprsPtrByValue(input));
      for (int tensor_axis = 0; tensor_axis < dim_exprs_ptr->size();
           ++tensor_axis) {
        const auto& dim_expr = dim_exprs_ptr->at(tensor_axis);
        map->emplace(dim_expr, InputDimIndex{in_idx, tensor_axis});
      }
    }
    return infer_meta_ctx;
  }

  adt::Result<const std::vector<symbol::DimExpr>*> GetShapeDimExprsPtrByValue(
      pir::Value value) const {
    auto* op = value.defining_op();
    ADT_CHECK(op != nullptr);
    auto* program = op->GetParentProgram();
    auto& shape_analysis = ::pir::ShapeAnalysisManager::Instance().Get(program);
    const auto& shape_or_data = shape_analysis.GetShapeOrDataForValue(value);
    using RetT = adt::Result<const std::vector<symbol::DimExpr>*>;
    return shape_or_data.Match(
        [&](const symbol::TensorShapeOrDataDimExprs& impl) -> RetT {
          return &impl.shape();
        },
        [&](const auto&) -> RetT {
          return adt::errors::TypeError{
              "GetShapeDimExprsPtrByValue only support "
              "TensorShapeOrDataDimExprs."};
        });
  }

  adt::Result<AnfExpr> ConstructDtype(ap::axpr::LetContext* ctx,
                                      const OpInferMetaCtx& infer_meta_ctx,
                                      pir::Type type) const {
    try {
      ::phi::DataType phi_dtype = ::paddle::dialect::TransToPhiDataType(type);
      ADT_LET_CONST_REF(dtype, ap::axpr::GetDataTypeFromPhiDataType(phi_dtype));
      return static_cast<AnfExpr>(ctx->Var("DataType").Attr(dtype.Name()));
    } catch (const std::exception& e) {
      return adt::errors::TypeError{
          "failed to cast from pir data type to phi data type."};
    }
  }

  adt::Result<AnfExpr> ConstructDDims(
      ap::axpr::LetContext* ctx,
      const OpInferMetaCtx& infer_meta_ctx,
      const std::vector<symbol::DimExpr>& dim_exprs) const {
    std::vector<AnfExpr> anf_dims;
    for (const auto& dim_expr : dim_exprs) {
      const auto& anf_dim_expr =
          ConstructDDimDimExpr(ctx, infer_meta_ctx, dim_expr);
      if (anf_dim_expr.HasOkValue()) {
        anf_dims.emplace_back(anf_dim_expr.GetOkValue());
      } else if (anf_dim_expr.GetError()
                     .template Has<adt::errors::MismatchError>()) {
        // TODO(lixinqi): anf_dims.emplace_back(AnfExpr{ctx->Int64(-1)});
        return anf_dim_expr.GetError();
      } else {
        return anf_dim_expr.GetError();
      }
    }
    return ctx->Call(ap::axpr::kBuiltinList(), anf_dims);
  }

  adt::Result<AnfExpr> ConstructDDimDimExpr(
      ap::axpr::LetContext* ctx,
      const OpInferMetaCtx& infer_meta_ctx,
      const symbol::DimExpr& dim_expr) const {
    return dim_expr.Match(
        [&](int64_t c) -> adt::Result<AnfExpr> { return ctx->Int64(c); },
        [&](const auto&) -> adt::Result<AnfExpr> {
          return ConstructDDimDimExprByInputs(ctx, infer_meta_ctx, dim_expr);
        });
  }

  adt::Result<AnfExpr> ConstructDDimDimExprByInputs(
      ap::axpr::LetContext* ctx,
      const OpInferMetaCtx& infer_meta_ctx,
      const symbol::DimExpr& dim_expr) const {
    const auto& idx_iter = infer_meta_ctx.dim_expr2in_dim_index.find(dim_expr);
    if (idx_iter == infer_meta_ctx.dim_expr2in_dim_index.end()) {
      return adt::errors::MismatchError{};
    }
    auto anf_expr_iter = infer_meta_ctx.dim_expr2anf_expr.find(dim_expr);
    if (anf_expr_iter == infer_meta_ctx.dim_expr2anf_expr.end()) {
      const auto& in_dim = ConstructInDimExpr(ctx, idx_iter->second);
      anf_expr_iter =
          infer_meta_ctx.dim_expr2anf_expr.emplace(dim_expr, in_dim).first;
    }
    return anf_expr_iter->second;
  }

  AnfExpr ConstructInDimExpr(ap::axpr::LetContext* ctx,
                             const InputDimIndex& idx) const {
    return static_cast<AnfExpr>(
        ctx->Var("inputs").At(idx.input_idx).Attr("dims").At(idx.tensor_axis));
  }

  adt::Result<AnfExpr> GetCodeFromBuiltinSerializableAttrMap(
      ap::axpr::LetContext* ctx,
      const ap::axpr::AttrMap<ap::axpr::SerializableValue>& attr_map) const {
    return ap::axpr::BuiltinSerializableAttrMapToAxprHelper{}.Convert(ctx,
                                                                      attr_map);
  }

  adt::Result<std::string> GetKernelDispatchConstDataLambdaStr(
      const DrrPackedIrOp& res_ptn_ir_op,
      const GraphMatchCtx& match_ctx,
      const ap::axpr::AttrMap<ap::axpr::SerializableValue>&
          kernel_dispatch_const_data) const {
    ap::axpr::LambdaExprBuilder lmbd;
    auto ConstructLambdaBody = [&](auto& ctx) -> adt::Result<AnfExpr> {
      ADT_LET_CONST_REF(data,
                        GetCodeFromBuiltinSerializableAttrMap(
                            &ctx, kernel_dispatch_const_data));
      return data;
    };
    ADT_LET_CONST_REF(anf_expr, lmbd.TryLambda({}, ConstructLambdaBody));
    return anf_expr.DumpToJsonString();
  }

  struct SerializedCodeGenResult {
    std::string code_gen_lambda_str;
    ap::axpr::Function<ap::axpr::SerializableValue> kernel_dispatch_func;
    ap::axpr::AttrMap<ap::axpr::SerializableValue> kernel_dispatch_const_data;
  };

  adt::Result<CodeGenResult> CodeGen(const DrrPackedIrOp& res_ptn_ir_op,
                                     const GraphMatchCtx& match_ctx) const {
    const auto& op_declare = res_ptn_ir_op->op_declare;
    ADT_LET_CONST_REF(
        op_declare_data,
        op_declare->cast_data<ap::drr::ResPtnPackedIrOpDeclareData>());
    const auto& lambda = op_declare_data->code_gen_func();
    ADT_LET_CONST_REF(code_gen_result,
                      GetApKernelModule(lambda, match_ctx, res_ptn_ir_op));
    const auto& kernel_dispatch_func = code_gen_result->kernel_dispatch_func;
    auto* data = &code_gen_result.shared_ptr()->kernel_dispatch_const_data;
    ADT_RETURN_IF_ERR(InsertOrCheckApKernelInputIndexOrSlices(
        data, res_ptn_ir_op, match_ctx));
    ADT_RETURN_IF_ERR(InsertOrCheckApKernelOutputIndexOrSlices(
        data, res_ptn_ir_op, match_ctx));
    ADT_RETURN_IF_ERR(
        InsertOrCheckApKernelInputName2Index(data, res_ptn_ir_op, match_ctx));
    ADT_RETURN_IF_ERR(
        InsertOrCheckApKernelOutputName2Index(data, res_ptn_ir_op, match_ctx));
    return code_gen_result;
  }

  adt::Result<adt::Ok> InsertOrCheckApKernelInputIndexOrSlices(
      ap::axpr::AttrMap<ap::axpr::SerializableValue>* object,
      const DrrPackedIrOp& res_ptn_ir_op,
      const GraphMatchCtx& match_ctx) const {
    adt::List<ap::axpr::SerializableValue> list;
    using Ok = adt::Result<adt::Ok>;
    auto DoEachIndex = [&](int64_t idx) -> Ok {
      list->emplace_back(idx);
      return adt::Ok{};
    };
    auto DoEachSlice = [&](int64_t start, int64_t end) -> Ok {
      adt::List<ap::axpr::SerializableValue> range{start, end};
      list->emplace_back(range);
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(VisitApKernelInputIndexOrSlice(
        res_ptn_ir_op, match_ctx, DoEachIndex, DoEachSlice));
    const std::string key{"__builtin_ap_kernel_input_indexes_slices"};
    if ((*object)->Has(key)) {
      ADT_LET_CONST_REF(old_list, (*object)->Get(key));
      ADT_CHECK(ap::axpr::SerializableValue{list} == old_list);  // NOLINT
    } else {
      ADT_CHECK((*object)->Emplace(key, list));
    }
    return adt::Ok{};
  }

  adt::Result<adt::Ok> InsertOrCheckApKernelOutputIndexOrSlices(
      ap::axpr::AttrMap<ap::axpr::SerializableValue>* object,
      const DrrPackedIrOp& res_ptn_ir_op,
      const GraphMatchCtx& match_ctx) const {
    adt::List<ap::axpr::SerializableValue> list;
    using Ok = adt::Result<adt::Ok>;
    auto DoEachIndex = [&](int64_t idx) -> Ok {
      list->emplace_back(idx);
      return adt::Ok{};
    };
    auto DoEachSlice = [&](int64_t start, int64_t end) -> Ok {
      adt::List<ap::axpr::SerializableValue> range{start, end};
      list->emplace_back(range);
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(VisitApKernelOutputIndexOrSlice(
        res_ptn_ir_op, match_ctx, DoEachIndex, DoEachSlice));
    const std::string key{"__builtin_ap_kernel_output_indexes_slices"};
    if ((*object)->Has(key)) {
      ADT_LET_CONST_REF(old_list, (*object)->Get(key));
      ADT_CHECK(ap::axpr::SerializableValue{list} == old_list);  // NOLINT
    } else {
      ADT_CHECK((*object)->Emplace(key, list));
    }
    return adt::Ok{};
  }

  adt::Result<CodeGenResult> GetApKernelModule(
      const ap::axpr::Value& lambda,
      const GraphMatchCtx& match_ctx,
      const DrrPackedIrOp& res_ptn_ir_op) const {
    ADT_LET_CONST_REF(src_ptn_ctx, ctx_.drr_ctx->GetSourcePatternCtx());
    IrMatchCtx ir_match_ctx{src_ptn_ctx, match_ctx};
    ADT_LET_CONST_REF(arg_source_ctx,
                      MakeArgSourceCtx(match_ctx, res_ptn_ir_op));
    CodeGenCtx code_gen_ctx{ir_match_ctx, res_ptn_ir_op, arg_source_ctx};
    ApKernelDefineHelper helper{ctx_.drr_ctx->circlable_ref_list};
    ADT_LET_CONST_REF(result, helper.Interpret(lambda, code_gen_ctx));
    return result;
  }

  adt::Result<ap::code_gen::ArgSourceCtx<PirNode>> MakeArgSourceCtx(
      const GraphMatchCtx& match_ctx,
      const DrrPackedIrOp& res_ptn_ir_op) const {
    ap::code_gen::MatchedResultPatternHelper<PirNode> helper{match_ctx,
                                                             ctx_.drr_ctx};
    ap::code_gen::ArgSourceMaker<PirNode> maker{helper};
    ADT_LET_CONST_REF(arg_source_ctx, maker.MakeArgSourceCtx(res_ptn_ir_op));
    return arg_source_ctx;
  }

  adt::Result<AnfExpr> ConvertApKernelModuleToAnfExpr(
      const CodeModule& m) const {
    return ap::code_module::ModuleToAxprHelper{}.ConvertModuleToAnfExpr(m);
  }

  adt::Result<std::string> GetKernelDispatchLambdaStr(
      const ap::axpr::Function<ap::axpr::SerializableValue>&
          kernel_dispatch_func) const {
    const auto& lambda = kernel_dispatch_func->lambda;
    ap::axpr::AnfExpr anf_expr = ap::axpr::ConvertCoreExprToAnfExpr(lambda);
    return anf_expr.DumpToJsonString();
  }

  adt::Result<pir::Value> MakeApPatternFusionOp(
      pir::PatternRewriter* rewriter,
      pir::Value input,
      std::size_t num_outputs,
      const std::string& code_gen_lambda_str,
      const std::string& infer_meta_lambda_str,
      const std::string& kernel_dispatch_lambda_str,
      const std::string& kernel_dispatch_const_data_lambda_str) const {
    auto ap_variadic = rewriter->Build<paddle::dialect::ApVariadicOp>(
        input,
        num_outputs,
        code_gen_lambda_str,
        infer_meta_lambda_str,
        kernel_dispatch_lambda_str,
        kernel_dispatch_const_data_lambda_str);
    return ap_variadic.out();
  }

  adt::Result<std::vector<pir::Value>> GetPackedOpOutputValues(
      pir::PatternRewriter* rewriter, pir::Value combined_out) const {
    auto split_op = rewriter->Build<pir::SplitOp>(combined_out);
    return split_op.outputs();
  }

  template <typename IrOpT>
  adt::Result<adt::Ok> UpdateConstructedOpOutputsInReplaceCtx(
      const GraphMatchCtx& match_ctx,
      const std::vector<pir::Value>& output_values,
      const IrOpT& res_ptn_ir_op,
      RewriteCtx* rewrite_ctx) const {
    auto UpdateRewriteCtx = [&](const DrrIrValue& ir_value,
                                const std::vector<pir::Value>& output_slice)
        -> adt::Result<adt::Ok> {
      return ir_value.Match(
          [&](const DrrNativeIrValue& ir_value) -> adt::Result<adt::Ok> {
            ADT_CHECK(output_slice.size() == 1);
            const auto& k = ir_value->name;
            const auto& v = output_slice.at(0);
            ADT_CHECK(rewrite_ctx->name2native_value.emplace(k, v).second);
            return adt::Ok{};
          },
          [&](const DrrPackedIrValue& ir_value) -> adt::Result<adt::Ok> {
            const auto& k = ir_value->name;
            const auto& v = output_slice;
            ADT_CHECK(rewrite_ctx->name2packed_values.emplace(k, v).second);
            return adt::Ok{};
          });
    };
    ADT_RETURN_IF_ERR(VisitEachMatchedDrrIrValueAndOutputSlice(
        match_ctx, output_values, res_ptn_ir_op, UpdateRewriteCtx));
    return adt::Ok{};
  }

  template <typename IrOpT, typename YieldT>
  adt::Result<adt::Ok> VisitEachMatchedDrrIrValueAndOutputSlice(
      const GraphMatchCtx& match_ctx,
      const std::vector<pir::Value>& output_values,
      const IrOpT& res_ptn_ir_op,
      const YieldT& Yield) const {
    ap::code_gen::MatchedResultPatternHelper<PirNode> helper{match_ctx,
                                                             ctx_.drr_ctx};
    return helper.VisitEachMatchedDrrIrValueAndOutputSlice<pir::Value>(
        output_values, res_ptn_ir_op, Yield);
  }

  adt::Result<std::size_t> GetResPtnNumPirValues(
      const DrrIrValue& drr_ir_value, const GraphMatchCtx& match_ctx) const {
    ap::code_gen::MatchedResultPatternHelper<PirNode> helper{match_ctx,
                                                             ctx_.drr_ctx};
    return helper.GetResPtnNumBirValues(drr_ir_value);
  }

  adt::Result<std::size_t> GetApKernelNumOutputs(
      const DrrPackedIrOp& res_ptn_ir_op,
      const GraphMatchCtx& match_ctx) const {
    ap::code_gen::MatchedResultPatternHelper<PirNode> helper{match_ctx,
                                                             ctx_.drr_ctx};
    return helper.GetApKernelNumOutputs(res_ptn_ir_op);
  }

  template <typename DoEachIndexT, typename DoEachSliceT>
  adt::Result<adt::Ok> VisitApKernelInputIndexOrSlice(
      const DrrPackedIrOp& res_ptn_ir_op,
      const GraphMatchCtx& match_ctx,
      const DoEachIndexT& DoEachIndex,
      const DoEachSliceT& DoEachSlice) const {
    ap::code_gen::MatchedResultPatternHelper<PirNode> helper{match_ctx,
                                                             ctx_.drr_ctx};
    return helper.VisitApKernelInputIndexOrSlice(
        res_ptn_ir_op, DoEachIndex, DoEachSlice);
  }

  template <typename DoEachIndexT, typename DoEachSliceT>
  adt::Result<adt::Ok> VisitApKernelOutputIndexOrSlice(
      const DrrPackedIrOp& res_ptn_ir_op,
      const GraphMatchCtx& match_ctx,
      const DoEachIndexT& DoEachIndex,
      const DoEachSliceT& DoEachSlice) const {
    ap::code_gen::MatchedResultPatternHelper<PirNode> helper{match_ctx,
                                                             ctx_.drr_ctx};
    return helper.VisitApKernelOutputIndexOrSlice(
        res_ptn_ir_op, DoEachIndex, DoEachSlice);
  }

  adt::Result<adt::Ok> InsertOrCheckApKernelInputName2Index(
      ap::axpr::AttrMap<ap::axpr::SerializableValue>* object,
      const DrrPackedIrOp& res_ptn_ir_op,
      const GraphMatchCtx& match_ctx) const {
    ap::axpr::AttrMap<ap::axpr::SerializableValue> name2idx;
    int64_t idx = 0;
    auto DoEachIrValue =
        [&](const DrrIrValue& drr_ir_value) -> adt::Result<adt::Ok> {
      ADT_CHECK(name2idx->Emplace(drr_ir_value.name(), idx));
      ++idx;
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(
        VisitResPtnInputIrValueByResPtnIrOp(res_ptn_ir_op, DoEachIrValue));
    const std::string key{"__builtin_ap_kernel_input_name_to_index"};
    if ((*object)->Has(key)) {
      ADT_LET_CONST_REF(old_name2idx_val, (*object)->Get(key));
      ADT_LET_CONST_REF(old_name2idx,
                        old_name2idx_val.template TryGet<
                            ap::axpr::AttrMap<ap::axpr::SerializableValue>>());
      ADT_CHECK(old_name2idx->storage == name2idx->storage);
    } else {
      ADT_CHECK((*object)->Emplace(key, name2idx));
    }
    return adt::Ok{};
  }

  adt::Result<adt::Ok> InsertOrCheckApKernelOutputName2Index(
      ap::axpr::AttrMap<ap::axpr::SerializableValue>* object,
      const DrrPackedIrOp& res_ptn_ir_op,
      const GraphMatchCtx& match_ctx) const {
    ap::axpr::AttrMap<ap::axpr::SerializableValue> name2idx;
    int64_t idx = 0;
    auto DoEachIrValue =
        [&](const DrrIrValue& drr_ir_value) -> adt::Result<adt::Ok> {
      ADT_CHECK(name2idx->Emplace(drr_ir_value.name(), idx));
      ++idx;
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(
        VisitResPtnOutputIrValueByResPtnIrOp(res_ptn_ir_op, DoEachIrValue));
    const std::string key{"__builtin_ap_kernel_output_name_to_index"};
    if ((*object)->Has(key)) {
      ADT_LET_CONST_REF(old_name2idx_val, (*object)->Get(key));
      ADT_LET_CONST_REF(old_name2idx,
                        old_name2idx_val.template TryGet<
                            ap::axpr::AttrMap<ap::axpr::SerializableValue>>());
      ADT_CHECK(old_name2idx->storage == name2idx->storage);
    } else {
      ADT_CHECK((*object)->Emplace(key, name2idx));
    }
    return adt::Ok{};
  }

  adt::Result<pir::Value> InsertCombinedOp(
      pir::PatternRewriter* rewriter,
      const std::vector<pir::Value>& inputs) const {
    auto combined_op = rewriter->Build<pir::CombineOp>(inputs);
    return combined_op.out();
  }

  adt::Result<adt::Ok> TrySetInsertPointer(
      pir::PatternRewriter* rewriter,
      const RewriteCtx& rewrite_ctx,
      const std::vector<pir::Value>& input_values,
      const GraphMatchCtx& match_ctx) const {
    ADT_LET_CONST_REF(opt_last_pir_op,
                      GetLastInputPirOp(rewriter->block(), input_values));
    if (opt_last_pir_op.has_value()) {
      rewriter->SetInsertionPointAfter(opt_last_pir_op.value());
    } else {
      ADT_RETURN_IF_ERR(
          SetDefaultInsertPointer(rewriter, rewrite_ctx, match_ctx));
    }
    return adt::Ok{};
  }

  adt::Result<std::optional<pir::Operation*>> GetLastInputPirOp(
      pir::Block* block, const std::vector<pir::Value>& input_values) const {
    const auto& ops = [&] {
      std::unordered_set<pir::Operation*> ret;
      for (const auto& value : input_values) {
        if (!value) {
          continue;
        }
        if (value.defining_op() != nullptr &&
            value.defining_op()->GetParent() == block) {
          ret.insert(value.defining_op());
        }
      }
      return ret;
    }();
    ADT_LET_CONST_REF(input_op2order_value, MakeMatchedOp2OrderValue(ops));
    auto OptOrderValue4Op = [&](pir::Operation* op) -> std::optional<int64_t> {
      const auto iter = input_op2order_value.find(op);
      if (iter == input_op2order_value.end()) {
        return std::nullopt;
      }
      return iter->second;
    };
    std::optional<pir::Operation*> last_op;
    std::optional<int64_t> op_order_value;
    for (auto* op : ops) {
      const auto& order_value = OptOrderValue4Op(op);
      if (!order_value.has_value()) {
        continue;
      }
      if (!op_order_value.has_value() ||
          op_order_value.value() < order_value.value()) {
        op_order_value = order_value.value();
        last_op = op;
      }
    }
    return last_op;
  }

  adt::Result<adt::Ok> SetDefaultInsertPointer(
      pir::PatternRewriter* rewriter,
      const RewriteCtx& rewrite_ctx,
      const GraphMatchCtx& match_ctx) const {
    ADT_LET_CONST_REF(last_pir_op, GetLastMatchedPirOp(rewrite_ctx, match_ctx));
    rewriter->SetInsertionPointAfter(last_pir_op);
    return adt::Ok{};
  }

  adt::Result<pir::Operation*> GetLastMatchedPirOp(
      const RewriteCtx& rewrite_ctx, const GraphMatchCtx& match_ctx) const {
    std::optional<pir::Operation*> last_op;
    std::optional<int64_t> op_order_value;
    auto UpdatePirOp = [&](pir::Operation* op) -> adt::Result<adt::Ok> {
      ADT_LET_CONST_REF(order_value, rewrite_ctx.GetMatchedOpOrderValue(op));
      if (!op_order_value.has_value() || op_order_value.value() < order_value) {
        op_order_value = order_value;
        last_op = op;
      }
      return adt::Ok{};
    };
    auto UpdateLastOp = [&](const DrrGraphNode& op) -> adt::Result<adt::Ok> {
      ADT_LET_CONST_REF(pir_node, match_ctx->GetSoleBigGraphNode(op));
      return pir_node.Match(
          [&](const ap::paddle::NativeIrOp& ir_op) -> adt::Result<adt::Ok> {
            return UpdatePirOp(ir_op.op);
          },
          [&](const ap::paddle::PackedIrOp& ir_op) -> adt::Result<adt::Ok> {
            return UpdatePirOp(ir_op.fusion_op);
          },
          [](const auto&) -> adt::Result<adt::Ok> { return adt::Ok{}; });
    };
    ADT_RETURN_IF_ERR(match_ctx->VisitSmallGraphNode(UpdateLastOp));
    ADT_CHECK(last_op.has_value());
    return last_op.value();
  }

  template <typename IrOpT, typename YieldT>
  adt::Result<adt::Ok> VisitResPtnInputIrValueByResPtnIrOp(
      const IrOpT& res_ptn_ir_op, const YieldT& Yield) const {
    ap::drr::ResultPatternHelper helper{ctx_.drr_ctx};
    return helper.VisitResPtnInputIrValueByResPtnIrOp(res_ptn_ir_op, Yield);
  }

  template <typename IrOpT, typename YieldT>
  adt::Result<adt::Ok> VisitResPtnOutputIrValueByResPtnIrOp(
      const IrOpT& res_ptn_ir_op, const YieldT& Yield) const {
    ap::drr::ResultPatternHelper helper{ctx_.drr_ctx};
    return helper.VisitResPtnOutputIrValueByResPtnIrOp(res_ptn_ir_op, Yield);
  }

  std::optional<DrrIrValue> SrcPtnIrValue4ResPtnIrValue(
      const DrrIrValue& res_ptn_ir_value) const {
    ap::drr::ResultPatternHelper helper{ctx_.drr_ctx};
    return helper.SrcPtnIrValue4ResPtnIrValue(res_ptn_ir_value);
  }

  template <typename IrOpT>
  adt::Result<adt::Ok> InsertInputPirValueToReplaceCtx(
      const IrOpT& res_ptn_ir_op,
      RewriteCtx* rewrite_ctx,
      const GraphMatchCtx& match_ctx) const {
    using Ok = adt::Result<adt::Ok>;
    auto InitInput = [&](const DrrIrValue& drr_ir_value) -> Ok {
      return drr_ir_value.Match(
          [&](const DrrNativeIrValue& res_ptn_ir_value) -> Ok {
            ADT_RETURN_IF_ERR(InsertNativeIrValueToReplaceCtx(
                res_ptn_ir_value, rewrite_ctx, match_ctx));
            return adt::Ok{};
          },
          [&](const DrrPackedIrValue& res_ptn_ir_value) -> Ok {
            ADT_RETURN_IF_ERR(InsertPackedIrValueToReplaceCtx(
                res_ptn_ir_value, rewrite_ctx, match_ctx));
            return adt::Ok{};
          });
    };
    ADT_RETURN_IF_ERR(
        VisitResPtnInputIrValueByResPtnIrOp(res_ptn_ir_op, InitInput));
    return adt::Ok{};
  }

  adt::Result<adt::Ok> InsertNativeIrValueToReplaceCtx(
      const DrrNativeIrValue& res_ptn_ir_value,
      RewriteCtx* rewrite_ctx,
      const GraphMatchCtx& match_ctx) const {
    const auto iter =
        rewrite_ctx->name2native_value.find(res_ptn_ir_value->name);
    if (iter != rewrite_ctx->name2native_value.end()) {
      return adt::Ok{};
    }
    const auto& opt_ir_value = SrcPtnIrValue4ResPtnIrValue(res_ptn_ir_value);
    ADT_CHECK(opt_ir_value.has_value());
    const auto& ir_value = opt_ir_value.value();
    ADT_LET_CONST_REF(pir_node,
                      match_ctx->GetSoleBigGraphNode(ir_value.node()));
    ADT_LET_CONST_REF(pir_value,
                      pir_node.template TryGet<ap::paddle::NativeIrValue>())
        << adt::errors::TypeError{
               "pir_node is not an ap::paddle::NativeIrValue"};
    rewrite_ctx->name2native_value[ir_value.name()] = pir_value.value;
    return adt::Ok{};
  }

  adt::Result<adt::Ok> InsertPackedIrValueToReplaceCtx(
      const DrrPackedIrValue& res_ptn_ir_value,
      RewriteCtx* rewrite_ctx,
      const GraphMatchCtx& match_ctx) const {
    using Ok = adt::Result<adt::Ok>;
    const auto iter =
        rewrite_ctx->name2packed_values.find(res_ptn_ir_value->name);
    if (iter != rewrite_ctx->name2packed_values.end()) {
      return adt::Ok{};
    }
    const auto& opt_ir_value = SrcPtnIrValue4ResPtnIrValue(res_ptn_ir_value);
    ADT_CHECK(opt_ir_value.has_value());
    const auto& ir_value = opt_ir_value.value();
    auto* vec = &rewrite_ctx->name2packed_values[ir_value.name()];
    ADT_CHECK(vec->empty());
    auto AppendNode = [&](const PirNode& pir_node) -> Ok {
      ADT_LET_CONST_REF(pir_value,
                        pir_node.template TryGet<ap::paddle::NativeIrValue>())
          << adt::errors::TypeError{
                 "pir_node is not an ap::paddle::NativeIrValue"};
      vec->emplace_back(pir_value.value);
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(
        match_ctx->VisitPackedBigGraphIrValueNode(ir_value.node(), AppendNode));
    return adt::Ok{};
  }

  adt::Result<PirNativeIrValue> CastToPirNativeIrValue(
      const PirNode& pir_node) const {
    using RetT = adt::Result<PirNativeIrValue>;
    return pir_node.Match(
        [&](const typename PirNode::native_value_type& bir_value) -> RetT {
          return bir_value;
        },
        [&](const typename PirNode::ref_value_type& ref_value) -> RetT {
          return ref_value.GetOwnerNativeIrValue();
        },
        [&](const auto&) -> RetT {
          return adt::errors::TypeError{
              "pir_node is not an PirNode::native_value_type or "
              "PirNode::ref_value_type"};
        });
  }

  adt::Result<std::vector<pir::Value>> GetMatchedPirInputsOfRestPtnPackedIrOp(
      const DrrPackedIrOp& res_ptn_ir_op,
      const GraphMatchCtx& match_ctx) const {
    std::vector<pir::Value> ret;
    auto CollectInput = [&](const PirNode& pir_node) -> adt::Result<adt::Ok> {
      ADT_LET_CONST_REF(pir_value, CastToPirNativeIrValue(pir_node));
      ret.emplace_back(pir_value.value);
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(VisitMatchedPirInputOfRestPtnPackedIrOp(
        res_ptn_ir_op, match_ctx, CollectInput));
    return ret;
  }

  template <typename YieldT>
  adt::Result<adt::Ok> VisitMatchedPirInputOfRestPtnPackedIrOp(
      const DrrPackedIrOp& res_ptn_ir_op,
      const GraphMatchCtx& match_ctx,
      const YieldT& Yield) const {
    ap::code_gen::MatchedResultPatternHelper<PirNode> helper{match_ctx,
                                                             ctx_.drr_ctx};
    return helper.VisitMatchedBirInputOfRestPtnPackedIrOp(res_ptn_ir_op, Yield);
  }

  adt::Result<std::vector<pir::Value>> GetMatchedPirOutputsOfRestPtnPackedIrOp(
      const DrrPackedIrOp& res_ptn_ir_op,
      const GraphMatchCtx& match_ctx) const {
    std::vector<pir::Value> ret;
    using Ok = adt::Result<adt::Ok>;
    auto CollectOutput = [&](const PirNode& pir_node) -> Ok {
      ADT_LET_CONST_REF(pir_value, CastToPirNativeIrValue(pir_node));
      ret.emplace_back(pir_value.value);
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(VisitMatchedPirOutputOfRestPtnPackedIrOp(
        res_ptn_ir_op, match_ctx, CollectOutput));
    return ret;
  }

  template <typename YieldT>
  adt::Result<adt::Ok> VisitMatchedPirOutputOfRestPtnPackedIrOp(
      const DrrPackedIrOp& res_ptn_ir_op,
      const GraphMatchCtx& match_ctx,
      const YieldT& Yield) const {
    ap::code_gen::MatchedResultPatternHelper<PirNode> helper{match_ctx,
                                                             ctx_.drr_ctx};
    return helper.VisitMatchedBirOutputOfRestPtnPackedIrOp(res_ptn_ir_op,
                                                           Yield);
  }

  adt::Result<std::vector<pir::Value>> GetNativeOpInputValues(
      const DrrNativeIrOp& res_ptn_ir_op, const RewriteCtx& rewrite_ctx) const {
    std::vector<pir::Value> ret;
    auto CollectValues = [&](pir::Value value) -> adt::Result<adt::Ok> {
      ret.push_back(value);
      return adt::Ok{};
    };
    auto VisitAndCollect =
        [&](const DrrIrValue& drr_ir_value) -> adt::Result<adt::Ok> {
      return VisitPirValueByIrValue(drr_ir_value, rewrite_ctx, CollectValues);
    };
    ADT_RETURN_IF_ERR(
        VisitResPtnInputIrValueByResPtnIrOp(res_ptn_ir_op, VisitAndCollect));
    return ret;
  }

  adt::Result<std::vector<pir::Value>> GetPackedOpInputValues(
      const DrrPackedIrOp& res_ptn_ir_op, const RewriteCtx& rewrite_ctx) const {
    std::vector<pir::Value> ret;
    auto CollectValues = [&](pir::Value value) -> adt::Result<adt::Ok> {
      ret.push_back(value);
      return adt::Ok{};
    };
    auto VisitAndCollect =
        [&](const DrrIrValue& drr_ir_value) -> adt::Result<adt::Ok> {
      return VisitPirValueByIrValue(drr_ir_value, rewrite_ctx, CollectValues);
    };
    ADT_RETURN_IF_ERR(
        VisitResPtnInputIrValueByResPtnIrOp(res_ptn_ir_op, VisitAndCollect));
    return ret;
  }

  template <typename YieldT>
  adt::Result<adt::Ok> VisitPirValueByIrValue(const DrrIrValue& ir_value,
                                              const RewriteCtx& rewrite_ctx,
                                              const YieldT& Yield) const {
    ADT_RETURN_IF_ERR(ir_value.Match(
        [&](const DrrNativeIrValue& ir_value) -> adt::Result<adt::Ok> {
          const auto& name = ir_value->name;
          const auto& iter = rewrite_ctx.name2native_value.find(name);
          ADT_CHECK(iter != rewrite_ctx.name2native_value.end());
          return Yield(iter->second);
        },
        [&](const DrrPackedIrValue& ir_value) -> adt::Result<adt::Ok> {
          const auto& name = ir_value->name;
          const auto& iter = rewrite_ctx.name2packed_values.find(name);
          ADT_CHECK(iter != rewrite_ctx.name2packed_values.end());
          for (const auto& value : iter->second) {
            ADT_RETURN_IF_ERR(Yield(value));
          }
          return adt::Ok{};
        }));
    return adt::Ok{};
  }
};

struct ConstraintApplier {
  adt::Result<bool> Match(const DrrCtx& drr_ctx,
                          const GraphMatchCtx& graph_match_ctx) {
    if (NeedCheckExtraUse(drr_ctx)) {
      ADT_LET_CONST_REF(found_extra_use_for_tmps,
                        FindExtraUseForTmpValues(drr_ctx, graph_match_ctx));
      if (found_extra_use_for_tmps) {
        return false;
      }
    }
    return CheckConstraint(drr_ctx, graph_match_ctx);
  }

  bool NeedCheckExtraUse(const DrrCtx& drr_ctx) {
    if (!drr_ctx->drr_pass_type.has_value()) {
      return false;
    }
    return drr_ctx->drr_pass_type.value().Match(
        [&](const ap::drr::AbstractDrrPassType&) -> bool { return true; },
        [&](const ap::drr::AccessTopoDrrPassType&) -> bool { return false; },
        [&](const ap::drr::ReifiedDrrPassType&) -> bool { return true; });
  }

  adt::Result<bool> FindExtraUseForTmpValues(
      const DrrCtx& drr_ctx, const GraphMatchCtx& graph_match_ctx) {
    bool any_tmp_value_has_extra_use = false;
    auto FindExtraUseForTmpValue =
        [&](const DrrIrValue& ir_value) -> adt::Result<adt::LoopCtrl> {
      ADT_LET_CONST_REF(has_extra_use,
                        HasExtraUse(drr_ctx, graph_match_ctx, ir_value));
      if (has_extra_use) {
        any_tmp_value_has_extra_use = true;
        return adt::Break{};
      } else {
        return adt::Continue{};
      }
    };
    ADT_RETURN_IF_ERR(VisitEachSrcPtnTmpValue(
        drr_ctx, graph_match_ctx, FindExtraUseForTmpValue));
    return any_tmp_value_has_extra_use;
  }

  adt::Result<bool> HasExtraUse(const DrrCtx& drr_ctx,
                                const GraphMatchCtx& graph_match_ctx,
                                const DrrIrValue& ir_value) {
    bool has_extra_use = false;
    auto FindExtraUse =
        [&](const PirNode& downstream) -> adt::Result<adt::LoopCtrl> {
      ADT_LET_CONST_REF(matched, IsPirNodeMatched(graph_match_ctx, downstream));
      if (matched) {
        return adt::Continue{};
      } else {
        has_extra_use = true;
        return adt::Break{};
      }
    };
    ADT_RETURN_IF_ERR(
        VisitDownstreamPirNode(graph_match_ctx, ir_value, FindExtraUse));
    return has_extra_use;
  }

  adt::Result<bool> IsPirNodeMatched(const GraphMatchCtx& graph_match_ctx,
                                     const PirNode& downstream) {
    return graph_match_ctx->GetOptMatchedSmallGraphNode(downstream).has_value();
  }

  template <typename YieldT>
  adt::Result<adt::Ok> VisitDownstreamPirNode(
      const GraphMatchCtx& graph_match_ctx,
      const DrrIrValue& ir_value,
      const YieldT& Yield) {
    ADT_CHECK(ir_value.template Has<DrrNativeIrValue>());
    ADT_LET_CONST_REF(pir_node,
                      graph_match_ctx->GetSoleBigGraphNode(ir_value.node()));
    DefaultGraph<PirNode> pir_graph{};
    auto DoEachPirNode = [&](const PirNode& pir_node) -> adt::Result<adt::Ok> {
      ADT_RETURN_IF_ERR(Yield(pir_node));
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(pir_graph.VisitDownstreamNodes(pir_node, DoEachPirNode));
    return adt::Ok{};
  }

  template <typename YieldT>
  adt::Result<adt::Ok> VisitEachSrcPtnTmpValue(
      const DrrCtx& drr_ctx,
      const GraphMatchCtx& graph_match_ctx,
      const YieldT& Yield) {
    std::unordered_set<DrrIrValue> inputs;
    std::unordered_set<DrrIrValue> outputs;
    std::unordered_set<DrrIrValue> ir_values_reachable_to_outputs;
    ADT_RETURN_IF_ERR(GetSrcPtnIrValues(
        drr_ctx, &inputs, &outputs, &ir_values_reachable_to_outputs));
    for (const auto& input : inputs) {
      ADT_CHECK(ir_values_reachable_to_outputs.count(input) > 0)
          << adt::errors::ValueError{
                 "There are result pattern inputs not reachable to result "
                 "pattern outputs"};
    }
    for (const auto& ir_value : ir_values_reachable_to_outputs) {
      if (inputs.count(ir_value)) {
        continue;
      }
      if (outputs.count(ir_value)) {
        continue;
      }
      ADT_LET_CONST_REF(loop_ctrl, Yield(ir_value));
      if (loop_ctrl.template Has<adt::Break>()) {
        break;
      } else {
        ADT_CHECK(loop_ctrl.template Has<adt::Continue>());
      }
    }
    return adt::Ok{};
  }

  adt::Result<adt::Ok> GetSrcPtnIrValues(
      const DrrCtx& drr_ctx,
      std::unordered_set<DrrIrValue>* inputs,
      std::unordered_set<DrrIrValue>* outputs,
      std::unordered_set<DrrIrValue>* ir_values_reachable_to_outputs) {
    ADT_LET_CONST_REF(res_ptn_inputs, GetResPtnInputs(drr_ctx));
    ADT_LET_CONST_REF(res_ptn_outputs, GetResPtnOutputs(drr_ctx));
    ap::drr::ResultPatternHelper helper{drr_ctx};
    for (const auto& res_ptn_input : res_ptn_inputs) {
      const auto& src_ptn_node =
          helper.SrcPtnIrValue4ResPtnIrValue(res_ptn_input);
      if (src_ptn_node.has_value()) {
        inputs->insert(src_ptn_node.value());
      }
    }
    for (const auto& res_ptn_output : res_ptn_outputs) {
      const auto& src_ptn_node =
          helper.SrcPtnIrValue4ResPtnIrValue(res_ptn_output);
      if (src_ptn_node.has_value()) {
        outputs->insert(src_ptn_node.value());
      }
    }
    using YieldT = typename ap::adt::TopoWalker<DrrGraphNode>::NodeHandlerType;
    using Ok = adt::Result<adt::Ok>;
    ap::drr::DefaultDrrGraphDescriptor drr_graph{};
    auto ForEachNext = [&](const DrrGraphNode& node,
                           const YieldT& Yield) -> Ok {
      ADT_LET_CONST_REF(drr_node, node.Get());
      const auto& drr_value = DrrIrValue::OptCastFrom(drr_node);
      if (drr_value.has_value()) {
        if (outputs->count(drr_value.value())) {
          // `drr_value` is sink of the subgraph.
          return adt::Ok{};
        } else {
          return drr_graph.VisitDownstreamNodes(node, Yield);
        }
      } else {
        return drr_graph.VisitDownstreamNodes(node, Yield);
      }
    };
    auto ForEachPrev = [&](const DrrGraphNode& node,
                           const YieldT& Yield) -> Ok {
      ADT_LET_CONST_REF(drr_node, node.Get());
      const auto& drr_value = DrrIrValue::OptCastFrom(drr_node);
      if (drr_value.has_value()) {
        if (inputs->count(drr_value.value())) {
          // `drr_value` is sink of the subgraph.
          return adt::Ok{};
        } else {
          return drr_graph.VisitUpstreamNodes(node, Yield);
        }
      } else {
        return drr_graph.VisitUpstreamNodes(node, Yield);
      }
    };
    // Reversely walk subgraph
    ap::adt::TopoWalker<DrrGraphNode> reverse_graph(ForEachNext, ForEachPrev);
    std::list<DrrGraphNode> starts{};
    for (const auto& start : *outputs) {
      starts.emplace_back(start.node());
    }
    auto DoEachNode = [&](const DrrGraphNode& drr_graph_node) -> Ok {
      ADT_LET_CONST_REF(drr_node, drr_graph_node.Get());
      const auto& drr_ir_value = DrrIrValue::OptCastFrom(drr_node);
      if (drr_ir_value.has_value()) {
        ir_values_reachable_to_outputs->insert(drr_ir_value.value());
      }
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(reverse_graph(starts.begin(), starts.end(), DoEachNode));
    return adt::Ok{};
  }

  adt::Result<bool> CheckConstraint(const DrrCtx& drr_ctx,
                                    const GraphMatchCtx& graph_match_ctx) {
    if (!drr_ctx->constraint_func.has_value()) {
      return true;
    }
    const auto& constraint_func = drr_ctx->constraint_func.value();
    ADT_CHECK(drr_ctx->source_pattern_ctx.has_value());
    IrMatchCtx ir_match_ctx{drr_ctx->source_pattern_ctx.value(),
                            graph_match_ctx};
    const auto& args = GetConstraintFuncArgs(ir_match_ctx);
    ApDrrHelper ap_drr_helper{drr_ctx->circlable_ref_list};
    ADT_LET_CONST_REF(is_match_val,
                      ap_drr_helper.Interpret(constraint_func, args));
    ADT_CHECK(drr_ctx->pass_name.has_value());
    ADT_LET_CONST_REF(is_match, is_match_val.template CastTo<bool>())
        << adt::errors::TypeError{
               std::string() +
               "constraint function should return a bool (not " +
               ap::axpr::GetTypeName(is_match_val) +
               "). pass_name: " + drr_ctx->pass_name.value()};
    return is_match;
  }

  std::vector<ap::axpr::Value> GetConstraintFuncArgs(
      const IrMatchCtx& ir_match_ctx) {
    ap::ir_match::OpMatchCtx<PirNode> op_match_ctx{ir_match_ctx.shared_ptr()};
    ap::ir_match::TensorMatchCtx<PirNode> tensor_match_ctx{
        ir_match_ctx.shared_ptr()};
    return std::vector<ap::axpr::Value>{
        ap::ir_match::GetOpMatchCtxClass<ap::axpr::Value, PirNode>().New(
            op_match_ctx),
        ap::ir_match::GetTensorMatchCtxClass<ap::axpr::Value, PirNode>().New(
            tensor_match_ctx),
    };
  }
};

struct NativeOpAnchorApGenericDrrPatternMatcher {
  const ApGenericDrrPatternCtx& ctx_;

  using Self = NativeOpAnchorApGenericDrrPatternMatcher;

  static adt::Result<std::optional<GraphMatchCtx>> Match(const DrrCtx& drr_ctx,
                                                         pir::Operation* op) {
    ADT_LET_CONST_REF(pattern_ctx,
                      ApGenericDrrPatternCtx::MakeFromDrrCtx(
                          drr_ctx,
                          /*steps_limit=*/std::nullopt,
                          std::make_shared<NaiveDrrCtxProvider>(drr_ctx)));
    Self matcher{pattern_ctx};
    return matcher.GetMatchCtx(op);
  }

  adt::Result<std::optional<GraphMatchCtx>> GetMatchCtx(
      pir::Operation* op) const {
    DefaultGraph<DrrGraphNode> drr_graph{};
    DefaultGraph<PirNode> pir_graph{};
    auto* parent_block = op->GetParent();
    ADT_CHECK(parent_block != nullptr);
    auto* parent_op = parent_block->GetParentOp();
    ADT_CHECK(!parent_op->isa<cinn::dialect::FusionOp>());
    ADT_CHECK(ctx_.native_op_anchor.has_value());
    const auto& native_op_anchor = ctx_.native_op_anchor.value();
    {
      ADT_LET_CONST_REF(
          anchor_topo_cstr,
          drr_graph.GetSmallGraphNodeTopoCstr(native_op_anchor->node));
      ap::paddle::NativeIrOp native_ir_op{op};
      ADT_LET_CONST_REF(topo_satisfy_constraint,
                        pir_graph.TopoSatisfy(native_ir_op, anchor_topo_cstr));
      bool satisfy_constraint = topo_satisfy_constraint;
      if (satisfy_constraint) {
        ap::graph::NodeDescriptor<PirNode> node_descriptor{};
        ADT_LET_CONST_REF(attrs_satisfy_constraint,
                          node_descriptor.AttrsSatisfyIfBothAreOpsOrValues(
                              native_ir_op, native_op_anchor->node));
        satisfy_constraint = attrs_satisfy_constraint;
        if (!attrs_satisfy_constraint) {
          ap::graph::NodeDescriptor<DrrGraphNode> drr_node_descriptor{};
          ap::graph::NodeDescriptor<PirNode> pir_node_descriptor{};
          LOG(ERROR) << "pir_node_descriptor.AttrsSatisfyIfBothAreOpsOrValues()"
                        " test failed. drr_node: "
                     << drr_node_descriptor.DebugId(native_op_anchor->node)
                     << ", pir_node: "
                     << pir_node_descriptor.DebugId(native_ir_op);
        }
      } else {
        ap::graph::NodeDescriptor<DrrGraphNode> drr_node_descriptor{};
        ap::graph::NodeDescriptor<PirNode> pir_node_descriptor{};
        LOG(ERROR) << "pir_graph.TopoSatisfy() test failed. drr_node: "
                   << drr_node_descriptor.DebugId(native_op_anchor->node)
                   << ", pir_node: "
                   << pir_node_descriptor.DebugId(native_ir_op);
      }
      ADT_CHECK(satisfy_constraint) << adt::errors::ValueError{
          std::string() +
          "pir_graph.TopoSatisfy() or "
          "node_descriptor.AttrsSatisfyIfBothAreOpsOrValues() test failed. "
          "drr_pass_name: " +
          ctx_.drr_ctx->pass_name.value()};
    }
    ADT_LET_CONST_REF(drr_op_result_anchor,
                      GetFirstNativeDrrIrOpResult(native_op_anchor));
    ADT_LET_CONST_REF(pir_op_result_anchor, GetFirstNativePirIrOpResult(op));
    {
      ADT_LET_CONST_REF(
          drr_op_result_anchor_topo_cstr,
          drr_graph.GetSmallGraphNodeTopoCstr(drr_op_result_anchor));
      ADT_LET_CONST_REF(topo_satisfy_constraint,
                        pir_graph.TopoSatisfy(pir_op_result_anchor,
                                              drr_op_result_anchor_topo_cstr));
      bool satisfy_constraint = topo_satisfy_constraint;
      if (satisfy_constraint) {
        ap::graph::NodeDescriptor<PirNode> node_descriptor{};
        ADT_LET_CONST_REF(attrs_satisfy_constraint,
                          node_descriptor.AttrsSatisfyIfBothAreOpsOrValues(
                              pir_op_result_anchor, drr_op_result_anchor));
        satisfy_constraint = attrs_satisfy_constraint;
      }
      ADT_CHECK(satisfy_constraint) << adt::errors::ValueError{
          std::string() +
          "TopoSatisfy() or AttrsSatisfyIfBothAreOpsOrValues() "
          "test failed. pir_op_result_anchor: " +
          DebugId(pir_op_result_anchor) +
          ", drr_op_result_anchor: " + DebugId(drr_op_result_anchor) +
          ", pir_op: " + DebugId(ap::paddle::NativeIrOp{op}) +
          ", drr_native_op: " + DebugId(native_op_anchor->node) + "."};
    }
    std::optional<GraphMatchCtx> opt_graph_match_ctx;
    {
      NativeORGraph<PirNode> pir_native_operand_result_graph{};
      NativeORGraph<DrrGraphNode> drr_native_operand_result_graph{};
      using NativeOR = ap::drr::topo_kind::NativeOperandAndResult;
      ap::ir_match::GraphMatcher<PirNode, NativeOR, NativeOR> graph_matcher(
          pir_native_operand_result_graph, drr_native_operand_result_graph);
      ADT_LET_CONST_REF(graph_ctx,
                        graph_matcher.MatchByAnchor(pir_op_result_anchor,
                                                    drr_op_result_anchor));
      opt_graph_match_ctx = graph_ctx;
      ADT_LET_CONST_REF(graph_matched,
                        graph_matcher.IsGraphMatched(
                            opt_graph_match_ctx.value(), drr_op_result_anchor));
      ADT_CHECK(graph_matched) << adt::errors::MismatchError{};
    }
    ADT_CHECK(opt_graph_match_ctx.has_value());
    {
      ADT_LET_CONST_REF(
          ref_match_ctx,
          GetRefMatchCtx(opt_graph_match_ctx.value(), drr_op_result_anchor));
      RefAugmentedGraph<PirNode> pir_augmented_graph{ref_match_ctx};
      using RefAugmented = ap::drr::topo_kind::RefAugmented;
      using Default = ap::drr::topo_kind::Default;
      ap::ir_match::GraphMatcher<PirNode, RefAugmented, Default> graph_matcher(
          pir_augmented_graph, drr_graph);
      ADT_RETURN_IF_ERR(graph_matcher.UpdateByConnectionsUntilDone(
          &opt_graph_match_ctx.value(), drr_op_result_anchor));
      auto UpdateUntilDone = [&](auto* ctx) -> adt::Result<adt::LoopCtrl> {
        ADT_RETURN_IF_ERR(graph_matcher.UpdateByConnectionsUntilDone(
            ctx, drr_op_result_anchor));
        ADT_LET_CONST_REF(
            graph_matched,
            graph_matcher.IsGraphMatched(*ctx, drr_op_result_anchor));
        if (graph_matched) {
          return adt::Break{};
        } else {
          return adt::Continue{};
        }
      };
      ADT_RETURN_IF_ERR(graph_matcher.InplaceForcePickOneLastUndetermined(
          &opt_graph_match_ctx.value(), UpdateUntilDone));
      ADT_LET_CONST_REF(graph_matched,
                        graph_matcher.IsGraphMatched(
                            opt_graph_match_ctx.value(), drr_op_result_anchor));
      if (!graph_matched) {
        opt_graph_match_ctx = std::nullopt;
      }
    }
    if (!opt_graph_match_ctx.has_value()) {
      return std::nullopt;
    }
    ADT_LET_CONST_REF(
        match,
        ConstraintApplier{}.Match(ctx_.drr_ctx, opt_graph_match_ctx.value()));
    if (!match) {
      return std::nullopt;
    }
    return opt_graph_match_ctx;
  }

  std::string DebugId(const PirNode& pir_node) const {
    return ap::graph::NodeDescriptor<PirNode>{}.DebugId(pir_node);
  }

  std::string DebugId(const DrrGraphNode& drr_node) const {
    return ap::graph::NodeDescriptor<DrrGraphNode>{}.DebugId(drr_node);
  }

  template <typename NodeT>
  using AllOperandAndResultGraph =
      ap::graph::GraphDescriptor<NodeT,
                                 ap::drr::topo_kind::AllOperandAndResult>;

  using RefNodeInfo =
      ap::ir_match::RefNodeInfo<PirNativeIrValue, PirNativeIrOpOperand>;
  using RefMatchCtx =
      ap::ir_match::RefMatchCtx<PirNativeIrValue, PirNativeIrOpOperand>;

  adt::Result<RefMatchCtx> GetRefMatchCtx(const GraphMatchCtx& graph_match_ctx,
                                          const DrrGraphNode& anchor) const {
    AllOperandAndResultGraph<PirNode> pir_graph{};
    AllOperandAndResultGraph<DrrGraphNode> drr_graph{};
    using AllOR = ap::drr::topo_kind::AllOperandAndResult;
    ap::ir_match::GraphMatcher<PirNode, AllOR, AllOR> graph_matcher(pir_graph,
                                                                    drr_graph);
    RefMatchCtx ref_match_ctx{};
    using Ok = adt::Result<adt::Ok>;
    auto DoEachMismatched = [&](const DrrGraphNode& node) -> Ok {
      ADT_LET_CONST_REF(drr_node, node.Get());
      return drr_node.Match(
          [&](const DrrOptPackedIrOpResult& op_result) -> Ok {
            ADT_LET_CONST_REF(ref_node_info,
                              GetRefNodeInfo(graph_match_ctx, op_result));
            if (ref_node_info.has_value()) {
              ADT_RETURN_IF_ERR(
                  ref_match_ctx->AddRefNodeInfo(ref_node_info.value()));
            }
            return adt::Ok{};
          },
          [&](const DrrOptPackedIrOpOperand& impl) -> Ok {
            // do nothing.
            return adt::Ok{};
          },
          [&](const DrrPackedIrOpOperand& impl) -> Ok {
            // do nothing.
            return adt::Ok{};
          },
          [&](const DrrPackedIrOpResult& impl) -> Ok {
            // do nothing.
            return adt::Ok{};
          },
          [&](const auto& impl) -> Ok {
            const char* type_name = typeid(std::decay_t<decltype(impl)>).name();
            return adt::errors::ValueError{
                std::string() +
                "GetRefValue2Operands unexpected mismatched DrrGraphNode: " +
                type_name};
          });
    };
    ADT_RETURN_IF_ERR(graph_matcher.VisitMisMatchedNodes(
        graph_match_ctx, anchor, DoEachMismatched));
    return ref_match_ctx;
  }

  adt::Result<std::optional<RefNodeInfo>> GetRefNodeInfo(
      const GraphMatchCtx& graph_match_ctx,
      const DrrOptPackedIrOpResult& op_result) const {
    ADT_LET_CONST_REF(opt_inner_ref_node_info,
                      GetInnerRefNodeInfo(graph_match_ctx, op_result));
    if (opt_inner_ref_node_info.has_value()) {
      return opt_inner_ref_node_info.value();
    }
    ADT_LET_CONST_REF(opt_output_ref_node_info,
                      GetOutputRefNodeInfo(graph_match_ctx, op_result));
    if (opt_output_ref_node_info.has_value()) {
      return opt_output_ref_node_info.value();
    }
    ADT_LET_CONST_REF(opt_input_ref_node_info,
                      GetInputRefNodeInfo(graph_match_ctx, op_result));
    if (opt_input_ref_node_info.has_value()) {
      return opt_input_ref_node_info.value();
    }
    return std::nullopt;
  }

  adt::Result<std::optional<RefNodeInfo>> GetInnerRefNodeInfo(
      const GraphMatchCtx& graph_match_ctx,
      const DrrOptPackedIrOpResult& drr_op_result) const {
    DefaultGraph<PirNode> default_pir_graph{};
    AllOperandAndResultGraph<DrrGraphNode> all_o_r_drr_graph{};
    const auto& topo_match_ctx = graph_match_ctx->topo_match_ctx;
    ADT_LET_CONST_REF(
        drr_op_operand,
        all_o_r_drr_graph.CastSoleUnignoredInput<DrrOptPackedIrOpOperand>(
            drr_op_result));
    {
      ADT_LET_CONST_REF(num_drr_op_result_downstreams,
                        all_o_r_drr_graph.GetNumOutputs(drr_op_result));
      if (num_drr_op_result_downstreams == 0) {
        return std::nullopt;
      }
      ADT_LET_CONST_REF(num_drr_op_operand_upstreams,
                        all_o_r_drr_graph.GetNumInputs(drr_op_operand));
      if (num_drr_op_operand_upstreams == 0) {
        return std::nullopt;
      }
      ADT_CHECK(num_drr_op_operand_upstreams == 1);
    }
    ADT_LET_CONST_REF(
        drr_op_operand_upstream,
        all_o_r_drr_graph.CastSoleUnignoredInput<DrrNativeIrOpResult>(
            drr_op_operand));
    ADT_LET_CONST_REF(
        pir_op_operand_upstream,
        topo_match_ctx->GetSoleBigGraphNode(drr_op_operand_upstream->node));
    ADT_LET_CONST_REF(pir_native_ir_value,
                      CastPirSoleOutput<PirNativeIrValue>(
                          default_pir_graph, pir_op_operand_upstream));
    adt::List<PirNativeIrOpOperand> pir_op_operands{};
    {
      auto DoEachDownstream =
          [&](const DrrGraphNode& node) -> adt::Result<adt::Ok> {
        ADT_LET_CONST_REF(drr_node, node.Get());
        ADT_CHECK(drr_node.template Has<DrrNativeIrOpOperand>());
        ADT_LET_CONST_REF(pir_node, topo_match_ctx->GetSoleBigGraphNode(node));
        ADT_LET_CONST_REF(pir_native_ir_op_operand,
                          pir_node.TryGet<PirNativeIrOpOperand>());
        ADT_LET_CONST_REF(cur_pir_native_ir_value,
                          CastPirSoleInput<PirNativeIrValue>(
                              default_pir_graph, pir_native_ir_op_operand));
        if (cur_pir_native_ir_value == pir_native_ir_value) {
          pir_op_operands->push_back(pir_native_ir_op_operand);
        }
        return adt::Ok{};
      };
      ADT_RETURN_IF_ERR(all_o_r_drr_graph.VisitDownstreamNodes(
          drr_op_result->node, DoEachDownstream));
      if (pir_op_operands->empty()) {
        return std::nullopt;
      }
    }
    return RefNodeInfo{pir_native_ir_value, pir_op_operands};
  }

  adt::Result<std::optional<RefNodeInfo>> GetOutputRefNodeInfo(
      const GraphMatchCtx& graph_match_ctx,
      const DrrOptPackedIrOpResult& drr_op_result) const {
    DefaultGraph<DrrGraphNode> default_drr_graph{};
    DefaultGraph<PirNode> default_pir_graph{};
    AllOperandAndResultGraph<DrrGraphNode> all_o_r_drr_graph{};
    const auto& topo_match_ctx = graph_match_ctx->topo_match_ctx;
    ADT_LET_CONST_REF(
        drr_op_operand,
        all_o_r_drr_graph.CastSoleUnignoredInput<DrrOptPackedIrOpOperand>(
            drr_op_result));
    {
      ADT_LET_CONST_REF(num_drr_op_result_downstreams,
                        all_o_r_drr_graph.GetNumOutputs(drr_op_result));
      if (num_drr_op_result_downstreams != 0) {
        return std::nullopt;
      }
      ADT_LET_CONST_REF(num_drr_op_operand_upstreams,
                        all_o_r_drr_graph.GetNumInputs(drr_op_operand));
      if (num_drr_op_operand_upstreams == 0) {
        return std::nullopt;
      }
      ADT_CHECK(num_drr_op_operand_upstreams == 1);
    }
    ADT_LET_CONST_REF(
        drr_op_operand_upstream,
        all_o_r_drr_graph.CastSoleUnignoredInput<DrrNativeIrOpResult>(
            drr_op_operand));
    ADT_LET_CONST_REF(
        pir_op_operand_upstream,
        topo_match_ctx->GetSoleBigGraphNode(drr_op_operand_upstream->node));
    ADT_LET_CONST_REF(pir_native_ir_value,
                      CastPirSoleOutput<PirNativeIrValue>(
                          default_pir_graph, pir_op_operand_upstream));
    ADT_LET_CONST_REF(
        drr_ir_value,
        default_drr_graph.CastSoleUnignoredInput<DrrNativeIrValue>(
            drr_op_operand));
    std::unordered_set<PirNativeIrOpOperand> excluded;
    {
      auto DoEachDownstream =
          [&](const DrrGraphNode& node) -> adt::Result<adt::Ok> {
        ADT_LET_CONST_REF(drr_node, node.Get());
        if (!drr_node.template Has<DrrNativeIrOpOperand>()) {
          return adt::Ok{};
        }
        ADT_LET_CONST_REF(pir_node, topo_match_ctx->GetSoleBigGraphNode(node));
        ADT_LET_CONST_REF(pir_op_operand,
                          pir_node.TryGet<PirNativeIrOpOperand>());
        ADT_CHECK(excluded.emplace(pir_op_operand).second);
        return adt::Ok{};
      };
      ADT_RETURN_IF_ERR(default_drr_graph.VisitDownstreamNodes(
          drr_ir_value->node, DoEachDownstream));
    }
    adt::List<PirNativeIrOpOperand> pir_op_operands{};
    {
      auto DoEachDownstream = [&](const PirNode& node) -> adt::Result<adt::Ok> {
        if (!node.template Has<PirNativeIrOpOperand>()) {
          return adt::Ok{};
        }
        ADT_LET_CONST_REF(pir_op_operand, node.TryGet<PirNativeIrOpOperand>());
        if (excluded.count(pir_op_operand) == 0) {
          pir_op_operands->push_back(pir_op_operand);
        }
        return adt::Ok{};
      };
      ADT_RETURN_IF_ERR(default_pir_graph.VisitDownstreamNodes(
          pir_native_ir_value, DoEachDownstream));
    }
    return RefNodeInfo{pir_native_ir_value, pir_op_operands};
  }

  adt::Result<std::optional<RefNodeInfo>> GetInputRefNodeInfo(
      const GraphMatchCtx& graph_match_ctx,
      const DrrOptPackedIrOpResult& drr_op_result) const {
    DefaultGraph<PirNode> default_pir_graph{};
    AllOperandAndResultGraph<DrrGraphNode> all_o_r_drr_graph{};
    const auto& topo_match_ctx = graph_match_ctx->topo_match_ctx;
    ADT_LET_CONST_REF(
        drr_op_operand,
        all_o_r_drr_graph.CastSoleUnignoredInput<DrrOptPackedIrOpOperand>(
            drr_op_result));
    {
      ADT_LET_CONST_REF(num_drr_op_result_downstreams,
                        all_o_r_drr_graph.GetNumOutputs(drr_op_result));
      if (num_drr_op_result_downstreams == 0) {
        return std::nullopt;
      }
      ADT_LET_CONST_REF(num_drr_op_operand_upstreams,
                        all_o_r_drr_graph.GetNumInputs(drr_op_operand));
      if (num_drr_op_operand_upstreams != 0) {
        return std::nullopt;
      }
    }
    std::optional<PirNativeIrValue> pir_native_ir_value;
    adt::List<PirNativeIrOpOperand> pir_op_operands{};
    {
      auto DoEachDownstream =
          [&](const DrrGraphNode& node) -> adt::Result<adt::Ok> {
        ADT_LET_CONST_REF(drr_node, node.Get());
        ADT_CHECK(drr_node.template Has<DrrNativeIrOpOperand>());
        ADT_LET_CONST_REF(pir_node, topo_match_ctx->GetSoleBigGraphNode(node));
        ADT_LET_CONST_REF(pir_native_ir_op_operand,
                          pir_node.TryGet<PirNativeIrOpOperand>());
        ADT_LET_CONST_REF(cur_pir_native_ir_value,
                          CastPirSoleInput<PirNativeIrValue>(
                              default_pir_graph, pir_native_ir_op_operand));
        if (!pir_native_ir_value.has_value()) {
          ADT_LET_CONST_REF(
              cur_pir_native_ir_value_upstream,
              GetPirSoleInput(default_pir_graph, cur_pir_native_ir_value));
          if (!cur_pir_native_ir_value_upstream
                   .template Has<PirNativeIrOpResult>()) {
            return adt::Ok{};
          }
          pir_native_ir_value = cur_pir_native_ir_value;
        }
        ADT_CHECK(cur_pir_native_ir_value == pir_native_ir_value.value());
        pir_op_operands->push_back(pir_native_ir_op_operand);
        return adt::Ok{};
      };
      ADT_RETURN_IF_ERR(all_o_r_drr_graph.VisitDownstreamNodes(
          drr_op_result->node, DoEachDownstream));
    }
    if (!pir_native_ir_value.has_value()) {
      return std::nullopt;
    }
    if (pir_op_operands->empty()) {
      return std::nullopt;
    }
    return RefNodeInfo{pir_native_ir_value.value(), pir_op_operands};
  }

  template <typename PirNodeImplT, typename GraphT>
  adt::Result<PirNodeImplT> CastPirSoleOutput(const GraphT& pir_graph,
                                              const PirNode& node) const {
    std::optional<PirNodeImplT> opt_pir_node{};
    auto DoEachDownstream =
        [&](const PirNode& downstream) -> adt::Result<adt::Ok> {
      ADT_LET_CONST_REF(pir_node_impl, downstream.TryGet<PirNodeImplT>());
      ADT_CHECK(!opt_pir_node.has_value());
      opt_pir_node = pir_node_impl;
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(pir_graph.VisitDownstreamNodes(node, DoEachDownstream));
    ADT_CHECK(opt_pir_node.has_value());
    return opt_pir_node.value();
  }

  template <typename PirNodeImplT, typename GraphT>
  adt::Result<PirNodeImplT> CastPirSoleInput(const GraphT& pir_graph,
                                             const PirNode& node) const {
    std::optional<PirNodeImplT> opt_pir_node{};
    auto DoEachUpstream = [&](const PirNode& upstream) -> adt::Result<adt::Ok> {
      ADT_LET_CONST_REF(pir_node_impl, upstream.TryGet<PirNodeImplT>());
      ADT_CHECK(!opt_pir_node.has_value());
      opt_pir_node = pir_node_impl;
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(pir_graph.VisitUpstreamNodes(node, DoEachUpstream));
    ADT_CHECK(opt_pir_node.has_value());
    return opt_pir_node.value();
  }

  template <typename GraphT>
  adt::Result<PirNode> GetPirSoleInput(const GraphT& pir_graph,
                                       const PirNode& node) const {
    std::optional<PirNode> opt_pir_node{};
    auto DoEachUpstream = [&](const PirNode& upstream) -> adt::Result<adt::Ok> {
      ADT_CHECK(!opt_pir_node.has_value());
      opt_pir_node = upstream;
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(pir_graph.VisitUpstreamNodes(node, DoEachUpstream));
    ADT_CHECK(opt_pir_node.has_value());
    return opt_pir_node.value();
  }

  template <typename GraphT>
  adt::Result<std::size_t> GetNumPirOutputs(const GraphT& pir_graph,
                                            const PirNode& node) const {
    std::size_t num_outputs = 0;
    auto DoEachDownstream =
        [&](const PirNode& downstream) -> adt::Result<adt::Ok> {
      ++num_outputs;
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(pir_graph.VisitDownstreamNodes(node, DoEachDownstream));
    return num_outputs;
  }

  template <typename GraphT>
  adt::Result<std::size_t> GetNumPirInputs(const GraphT& pir_graph,
                                           const PirNode& node) const {
    std::size_t num_inputs = 0;
    auto DoEachUpstream = [&](const PirNode& upstream) -> adt::Result<adt::Ok> {
      ++num_inputs;
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(pir_graph.VisitUpstreamNodes(node, DoEachUpstream));
    return num_inputs;
  }

  adt::Result<DrrGraphNode> GetFirstNativeDrrIrOpResult(
      const DrrNativeIrOp& op) const {
    ADT_LET_CONST_REF(downstreams, op->node.DownstreamNodes());
    ADT_CHECK(downstreams.size() > 0);
    using List = adt::List<DrrGraphNode>;
    using Vec = ap::graph::IndexedTag<List>;
    ADT_LET_CONST_REF(indexed_list, downstreams.template TryGet<Vec>());
    return indexed_list.data->at(0);
  }

  adt::Result<PirNode> GetFirstNativePirIrOpResult(pir::Operation* op) const {
    ADT_CHECK(!op->isa<cinn::dialect::FusionOp>());
    ADT_CHECK(op->num_results() > 0);
    pir::Value value = op->result(0);
    ap::paddle::NativeIrOpResult ir_op_result{
        pir::OpResult::dyn_cast_from(value)};
    return ir_op_result;
  }
};

struct OpEraseHelepr {
  ap::drr::SourcePatternCtx source_pattern_ctx_;

  adt::Result<adt::Ok> EraseUnusedOps(pir::PatternRewriter* rewriter,
                                      const GraphMatchCtx& graph_match_ctx) {
    using Ok = adt::Result<adt::Ok>;
    auto TryErase = [&](const DrrGraphNode& drr_graph_node) -> Ok {
      ADT_LET_CONST_REF(drr_node, drr_graph_node.Get());
      return EraseIfUnused(drr_node, rewriter, graph_match_ctx);
    };
    ADT_RETURN_IF_ERR(ReversedVisitSrcPtnGraph(TryErase));
    return adt::Ok{};
  }

 private:
  adt::Result<adt::Ok> EraseIfUnused(const DrrNode& drr_node,
                                     pir::PatternRewriter* rewriter,
                                     const GraphMatchCtx& graph_match_ctx) {
    using Ok = adt::Result<adt::Ok>;
    return drr_node.Match(
        [&](const DrrNativeIrOp&) -> Ok {
          return EraseOpIfUnused(drr_node, rewriter, graph_match_ctx);
        },
        [&](const DrrPackedIrOp&) -> Ok {
          return EraseOpIfUnused(drr_node, rewriter, graph_match_ctx);
        },
        [&](const DrrOptPackedIrOp&) -> Ok {
          return EraseOpIfUnused(drr_node, rewriter, graph_match_ctx);
        },
        [&](const auto& impl) -> Ok {
          // Do nothing.
          return adt::Ok{};
        });
  }

  adt::Result<adt::Ok> EraseOpIfUnused(const DrrNode& drr_node,
                                       pir::PatternRewriter* rewriter,
                                       const GraphMatchCtx& graph_match_ctx) {
    ADT_LET_CONST_REF(pir_node,
                      graph_match_ctx->GetSoleBigGraphNode(drr_node.node()));
    using Ok = adt::Result<adt::Ok>;
    return pir_node.Match(
        [&](const ap::paddle::NativeIrOp& ir_op) -> Ok {
          return ErasePirOpIfUnused(ir_op.op, rewriter);
        },
        [&](const ap::paddle::PackedIrOp& ir_op) -> Ok {
          return ErasePirOpIfUnused(ir_op.fusion_op, rewriter);
        },
        [&](const auto&) -> Ok {
          // Do nothing.
          return adt::Ok{};
        });
  }

  adt::Result<adt::Ok> ErasePirOpIfUnused(const pir::Operation* op,
                                          pir::PatternRewriter* rewriter) {
    auto* mut_op = const_cast<pir::Operation*>(op);
    if (mut_op->use_empty()) {
      rewriter->EraseOp(mut_op);
    }
    return adt::Ok{};
  }

  template <typename YieldT>
  adt::Result<adt::Ok> ReversedVisitSrcPtnGraph(const YieldT& Yield) {
    std::list<DrrGraphNode> sinks;
    for (const auto& drr_node : source_pattern_ctx_->node_arena->nodes()) {
      const auto& drr_graph_node = drr_node.node();
      ADT_LET_CONST_REF(downstreams, drr_graph_node.DownstreamNodes());
      if (downstreams.size() == 0) {
        sinks.push_back(drr_graph_node);
      }
    }
    using Ok = adt::Result<adt::Ok>;
    ap::drr::DefaultDrrGraphDescriptor graph{};
    auto VisitPrev = [&](const DrrGraphNode& node, const auto& Yield) -> Ok {
      return graph.VisitDownstreamNodes(node, Yield);
    };
    auto VisitNext = [&](const DrrGraphNode& node, const auto& Yield) -> Ok {
      return graph.VisitUpstreamNodes(node, Yield);
    };
    ap::adt::TopoWalker<DrrGraphNode> walker{VisitPrev, VisitNext};
    ADT_RETURN_IF_ERR(walker(sinks.begin(), sinks.end(), Yield));
    return adt::Ok{};
  }
};

class NativeOpAnchorApGenericDrrPattern : public pir::RewritePattern {
 private:
  ApGenericDrrPatternCtx ctx_;
  ApRewriter ap_rewriter_;
  mutable std::size_t times_;

 public:
  NativeOpAnchorApGenericDrrPattern(pir::IrContext* ir_context,
                                    const ApGenericDrrPatternCtx& ctx)
      : pir::RewritePattern(ctx.anchor_op_name, 1, ir_context, {}),
        ctx_(ctx),
        times_(0),
        ap_rewriter_(ctx, &NativeOpAnchorApGenericDrrPatternMatcher::Match) {}

  bool MatchAndRewrite(
      pir::Operation* op,
      pir::PatternRewriter& rewriter) const override {  // // NOLINT
    if (ctx_.steps_limit.has_value()) {
      if (times_ >= ctx_.steps_limit.value()) {
        return false;
      }
    }
    const auto& ret = this->TryMatchAndRewrite(op, &rewriter);
    if (ret.HasError()) {
      VLOG(0) << "drr: " << ctx_.drr_ctx->pass_name.value() << " mismatched.";
      VLOG(6) << "\nTraceback (most recent call last):\n"
              << ret.GetError().CallStackToString() << "\n"
              << ret.GetError().class_name() << ": " << ret.GetError().msg()
              << "\npass_name: " << ctx_.drr_ctx->pass_name.value();
      return false;
    }
    bool success = ret.GetOkValue();
    if (success) {
      ++times_;
    }
    return success;
  }

  adt::Result<bool> TryMatchAndRewrite(pir::Operation* op,
                                       pir::PatternRewriter* rewriter) const {
    ADT_LET_CONST_REF(opt_match_ctx, GetMatchCtx(op));
    ADT_CHECK(ctx_.drr_ctx->pass_name.has_value());
    if (!opt_match_ctx.has_value()) {
      VLOG(0) << "drr: " << ctx_.drr_ctx->pass_name.value() << " mismatched.";
      return false;
    }
    VLOG(0) << "drr: " << ctx_.drr_ctx->pass_name.value() << " matched.";
    ADT_LET_CONST_REF(
        success, ap_rewriter_.Rewrite(opt_match_ctx.value(), op, rewriter));
    if (success) {
      ADT_CHECK(ctx_.drr_ctx->source_pattern_ctx.has_value());
      OpEraseHelepr erase_helper{ctx_.drr_ctx->source_pattern_ctx.value()};
      ADT_RETURN_IF_ERR(
          erase_helper.EraseUnusedOps(rewriter, opt_match_ctx.value()));
    }
    return success;
  }

  adt::Result<std::optional<GraphMatchCtx>> GetMatchCtx(
      pir::Operation* op) const {
    return NativeOpAnchorApGenericDrrPatternMatcher{ctx_}.GetMatchCtx(op);
  }
};

struct DefaultAnchorApGenericDrrPatternMatcher {
  const ApGenericDrrPatternCtx& ctx_;

  using Self = DefaultAnchorApGenericDrrPatternMatcher;

  static adt::Result<std::optional<GraphMatchCtx>> Match(const DrrCtx& drr_ctx,
                                                         pir::Operation* op) {
    ADT_LET_CONST_REF(pattern_ctx,
                      ApGenericDrrPatternCtx::MakeFromDrrCtx(
                          drr_ctx,
                          /*times_step=*/std::nullopt,
                          std::make_shared<NaiveDrrCtxProvider>(drr_ctx)));
    Self matcher{pattern_ctx};
    return matcher.GetMatchCtx(op);
  }

  adt::Result<std::optional<GraphMatchCtx>> GetMatchCtx(
      pir::Operation* op) const {
    auto* parent_block = op->GetParent();
    ADT_CHECK(parent_block != nullptr);
    auto* parent_op = parent_block->GetParentOp();
    ADT_CHECK(!parent_op->isa<cinn::dialect::FusionOp>());
    const auto& default_anchor = ctx_.default_anchor;
    using Default = ap::drr::topo_kind::Default;
    ap::graph::GraphDescriptor<PirNode, Default> pir_graph{};
    ap::graph::GraphDescriptor<DrrGraphNode, Default> src_ptn_graph{};
    ap::ir_match::GraphMatcher<PirNode, Default, Default> graph_matcher(
        pir_graph, src_ptn_graph);
    ADT_LET_CONST_REF(
        anchor_topo_cstr,
        src_ptn_graph.GetSmallGraphNodeTopoCstr(default_anchor.node()));
    const auto& obj_node = CastToPirNode(op);
    ADT_LET_CONST_REF(topo_satisfy_constraint,
                      pir_graph.TopoSatisfy(obj_node, anchor_topo_cstr));
    bool satisfy_constraint = topo_satisfy_constraint;
    if (satisfy_constraint) {
      ap::graph::NodeDescriptor<PirNode> node_descriptor{};
      ADT_LET_CONST_REF(attrs_satisfy_constraint,
                        node_descriptor.AttrsSatisfyIfBothAreOpsOrValues(
                            obj_node, default_anchor.node()));
      satisfy_constraint = attrs_satisfy_constraint;
    }
    ADT_CHECK(satisfy_constraint) << adt::errors::ValueError{
        "TopoSatisfy() or AttrsSatisfyIfBothAreOpsOrValues() test failed."};
    ADT_LET_CONST_REF(
        graph_match_ctx,
        graph_matcher.MatchByAnchor(obj_node, default_anchor.node()));
    ADT_LET_CONST_REF(
        graph_matched,
        graph_matcher.IsGraphMatched(graph_match_ctx, default_anchor.node()));
    if (!graph_matched) {
      return std::nullopt;
    }
    ADT_LET_CONST_REF(constraint_matched,
                      ConstraintApplier{}.Match(ctx_.drr_ctx, graph_match_ctx));
    if (!constraint_matched) {
      return std::nullopt;
    }
    return graph_match_ctx;
  }

  PirNode CastToPirNode(pir::Operation* op) const {
    if (op->isa<cinn::dialect::FusionOp>()) {
      ap::paddle::PackedIrOp ir_op{op->dyn_cast<cinn::dialect::FusionOp>()};
      return ir_op;
    } else {
      ap::paddle::NativeIrOp ir_op{op};
      return ir_op;
    }
  }
};

class DefaultAnchorApGenericDrrPattern : public pir::RewritePattern {
 private:
  ApGenericDrrPatternCtx ctx_;
  mutable std::size_t times_;
  ApRewriter ap_rewriter_;

 public:
  DefaultAnchorApGenericDrrPattern(pir::IrContext* ir_context,
                                   const ApGenericDrrPatternCtx& ctx)
      : pir::RewritePattern(ctx.anchor_op_name, 1, ir_context, {}),
        ctx_(ctx),
        times_(0),
        ap_rewriter_(ctx, &DefaultAnchorApGenericDrrPatternMatcher::Match) {}

  bool MatchAndRewrite(
      pir::Operation* op,
      pir::PatternRewriter& rewriter) const override {  // // NOLINT
    if (ctx_.steps_limit.has_value()) {
      if (times_ >= ctx_.steps_limit.value()) {
        return false;
      }
    }
    const auto& ret = this->TryMatchAndRewrite(op, &rewriter);
    if (ret.HasError()) {
      VLOG(0) << "drr: " << ctx_.drr_ctx->pass_name.value() << " mismatched.";
      VLOG(6) << "\nTraceback (most recent call last):\n"
              << ret.GetError().CallStackToString() << "\n"
              << ret.GetError().class_name() << ": " << ret.GetError().msg()
              << "\npass_name: " << ctx_.drr_ctx->pass_name.value();
      return false;
    }
    bool success = ret.GetOkValue();
    if (success) {
      ++times_;
    }
    return success;
  }

  adt::Result<bool> TryMatchAndRewrite(pir::Operation* op,
                                       pir::PatternRewriter* rewriter) const {
    ADT_LET_CONST_REF(opt_match_ctx, GetMatchCtx(op));
    if (!opt_match_ctx.has_value()) {
      return false;
    }
    ADT_CHECK(ctx_.drr_ctx->pass_name.has_value());
    VLOG(0) << "drr: " << ctx_.drr_ctx->pass_name.value() << " matched.";
    ADT_LET_CONST_REF(
        success, ap_rewriter_.Rewrite(opt_match_ctx.value(), op, rewriter));
    if (success) {
      ADT_CHECK(ctx_.drr_ctx->source_pattern_ctx.has_value());
      OpEraseHelepr erase_helper{ctx_.drr_ctx->source_pattern_ctx.value()};
      ADT_RETURN_IF_ERR(
          erase_helper.EraseUnusedOps(rewriter, opt_match_ctx.value()));
    }
    return success;
  }

  adt::Result<std::optional<GraphMatchCtx>> GetMatchCtx(
      pir::Operation* op) const {
    return DefaultAnchorApGenericDrrPatternMatcher{ctx_}.GetMatchCtx(op);
  }
};

class ApGenericDrrPass : public pir::PatternRewritePass {
 private:
  std::shared_ptr<DrrCtxProvider> drr_ctx_provider_;
  std::optional<int64_t> steps_limit_;

 public:
  explicit ApGenericDrrPass(
      const std::shared_ptr<DrrCtxProvider>& drr_ctx_provider,
      const std::string& name,
      std::optional<int64_t> steps_limit)
      : pir::PatternRewritePass(
            std::string() + "ap_lower_fusion_op_" + name + "_pass", 2),
        drr_ctx_provider_(drr_ctx_provider),
        steps_limit_(steps_limit) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext* context) override {
    pir::RewritePatternSet ps(context);
    const auto& ret = TryInitializePatterns(&ps, context);
    if (ret.HasError()) {
      LOG(ERROR) << "\nTraceback (most recent call last):\n"
                 << ret.GetError().CallStackToString() << "\n"
                 << "InitializePatterns " << ret.GetError().class_name() << ": "
                 << ret.GetError().msg();
    }
    return ps;
  }

  adt::Result<adt::Ok> TryInitializePatterns(pir::RewritePatternSet* ps,
                                             pir::IrContext* context) {
    auto AddFusionOpPattern = [&](const auto& drr_ctx) -> adt::Result<adt::Ok> {
      ADT_LET_CONST_REF(pattern_ctx,
                        ApGenericDrrPatternCtx::MakeFromDrrCtx(
                            drr_ctx, steps_limit_, drr_ctx_provider_));
      if (pattern_ctx.native_op_anchor.has_value()) {
        ps->Add(std::make_unique<NativeOpAnchorApGenericDrrPattern>(
            context, pattern_ctx));
      } else {
        ps->Add(std::make_unique<DefaultAnchorApGenericDrrPattern>(
            context, pattern_ctx));
      }
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(VisitEachDrrCtx(AddFusionOpPattern));
    return adt::Ok{};
  }

  template <typename YieldT>
  adt::Result<adt::Ok> VisitEachDrrCtx(const YieldT& Yield) {
    ADT_LET_CONST_REF(drr_ctx_list, drr_ctx_provider_->GetDrrCtxList());
    for (const auto& drr_ctx : *drr_ctx_list) {
      ADT_RETURN_IF_ERR(Yield(drr_ctx));
    }
    return adt::Ok{};
  }
};

class AbstractDrrCtxProvider : public DrrCtxProvider {
  std::weak_ptr<ap::memory::CirclableRefListBase> circlable_ref_list_;

 public:
  explicit AbstractDrrCtxProvider(
      const std::weak_ptr<ap::memory::CirclableRefListBase>& circlable_ref_list)
      : circlable_ref_list_(circlable_ref_list) {}

  adt::Result<adt::List<DrrCtx>> GetDrrCtxList() override {
    static adt::Result<adt::List<DrrCtx>> drr_ctx_list(MakeDrrCtxList());
    return drr_ctx_list;
  }

  adt::Result<adt::List<DrrCtx>> MakeDrrCtxList() {
    adt::List<DrrCtx> ret{};
    auto Collect = [&](const auto& drr_ctx) -> adt::Result<adt::Ok> {
      ret->emplace_back(drr_ctx);
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(VisitEachDrrCtxByAbstractDrrPassRegistryItems(Collect));
    return ret;
  }

  adt::Result<adt::Ok> PostProcess(
      adt::Result<std::optional<GraphMatchCtx>> (*Match)(const DrrCtx&,
                                                         pir::Operation* op),
      const DrrCtx& drr_ctx,
      pir::Operation* op,
      const GraphMatchCtx& match_ctx,
      const std::function<adt::Result<CodeGenResult>(const std::string&)>&
          CodeGenResult4FusedOpName) override {
    ap::reified_drr::ReifiedDrrPassDumpHelper dump_helper{};
    if (!dump_helper.DumpEnabled()) {
      return adt::Ok{};
    }
    ap::paddle::PirToAnfExprHelper attr2axpr_helper{};
    ADT_CHECK(drr_ctx->source_pattern_ctx.has_value());
    const auto& src_ptn_ctx = drr_ctx->source_pattern_ctx.value();
    ap::paddle::PirNodeMatchedSrcPtnCtxHelper src_ptn_ctx_helper(src_ptn_ctx,
                                                                 match_ctx);
    ADT_LET_CONST_REF(
        reified_drr_pass_class_lambda_anf_expr,
        dump_helper.Dump(
            /*abstract_drr_ctx=*/drr_ctx,
            /*attr2axpr_helper=*/&attr2axpr_helper,
            /*src_ptn_ctx_helper=*/&src_ptn_ctx_helper,
            /*CodeGenResult4FusedOpName=*/CodeGenResult4FusedOpName,
            /*nice=*/0));
    ADT_LET_CONST_REF(reified_drr_ctx,
                      GetReifiedDrrCtx(drr_ctx->circlable_ref_list,
                                       reified_drr_pass_class_lambda_anf_expr));
    ADT_LET_CONST_REF(opt_match_ctx, Match(reified_drr_ctx, op));
    ADT_CHECK(opt_match_ctx.has_value());
    return adt::Ok{};
  }

 private:
  static adt::Result<DrrCtx> GetReifiedDrrCtx(
      const std::weak_ptr<ap::memory::CirclableRefListBase>& circlable_ref_list,
      const ap::axpr::AnfExpr& reified_drr_pass_class_lambda_anf_expr) {
    const auto& core_expr = ap::axpr::ConvertAnfExprToCoreExpr(
        reified_drr_pass_class_lambda_anf_expr);
    using CoreExpr = ap::axpr::CoreExpr;
    ADT_LET_CONST_REF(atomic,
                      core_expr.template TryGet<ap::axpr::Atomic<CoreExpr>>());
    ADT_LET_CONST_REF(lambda,
                      atomic.template TryGet<ap::axpr::Lambda<CoreExpr>>());
    const auto& frames = ap::axpr::MakeBuiltinFrameAttrMap<ap::axpr::Value>();
    ap::axpr::CpsInterpreter interpreter{frames, circlable_ref_list};
    ADT_LET_CONST_REF(drr_pass_class_val, interpreter.Interpret(lambda, {}));
    ADT_LET_CONST_REF(
        drr_pass_class,
        drr_pass_class_val.template CastTo<
            ap::axpr::TypeImpl<ap::axpr::ClassInstance<ap::axpr::Value>>>());
    return ApDrrHelper{circlable_ref_list}.Interpret(
        drr_pass_class.class_attrs);
  }

  template <typename YieldT>
  adt::Result<adt::Ok> VisitEachDrrCtxByAbstractDrrPassRegistryItems(
      const YieldT& Yield) {
    ADT_LET_CONST_REF(registry, ApRegistryHelper{}.SingletonRegistry());
    const auto& abstract_drr_pass_registry_items =
        registry->abstract_drr_pass_registry_items;
    for (const auto& [abstract_drr_pass_name, nice2abstract_drr_pass_items] :
         abstract_drr_pass_registry_items) {
      std::optional<DrrCtx> opt_drr_ctx;
      for (const auto& [nice, abstract_drr_pass_items] :
           nice2abstract_drr_pass_items) {
        if (opt_drr_ctx.has_value()) {
          break;
        }
        for (const auto& abstract_drr_pass_item : abstract_drr_pass_items) {
          const auto& drr_ctx = GetDrrCtx(abstract_drr_pass_item);
          if (drr_ctx.HasOkValue()) {
            ADT_RETURN_IF_ERR(Yield(drr_ctx.GetOkValue()));
            opt_drr_ctx = drr_ctx.GetOkValue();
            break;
          } else {
            LOG(ERROR) << "\nTraceback (most recent call last):\n"
                       << drr_ctx.GetError().CallStackToString() << "\n"
                       << drr_ctx.GetError().class_name()
                       << ": abstract_drr_pass_name: " << abstract_drr_pass_name
                       << " nice: " << nice
                       << " msg: " << drr_ctx.GetError().msg();
          }
        }
      }
    }
    return adt::Ok{};
  }

  adt::Result<DrrCtx> GetDrrCtx(
      const ap::registry::AbstractDrrPassRegistryItem& abstract_drr_pass_item) {
    static ap::memory::Guard drr_ctx_mem_guard{};
    ADT_LET_CONST_REF(
        drr_ctx,
        ApDrrHelper{drr_ctx_mem_guard.circlable_ref_list()}.Interpret(
            abstract_drr_pass_item->cls));
    if (!drr_ctx->pass_name.has_value()) {
      drr_ctx.shared_ptr()->pass_name =
          abstract_drr_pass_item->abstract_drr_pass_name;
    }
    return drr_ctx;
  }
};

class ClassicDrrCtxProvider : public DrrCtxProvider {
  std::weak_ptr<ap::memory::CirclableRefListBase> circlable_ref_list_;

 public:
  explicit ClassicDrrCtxProvider(
      const std::weak_ptr<ap::memory::CirclableRefListBase>& circlable_ref_list)
      : circlable_ref_list_(circlable_ref_list) {}

  adt::Result<adt::List<DrrCtx>> GetDrrCtxList() override {
    static adt::Result<adt::List<DrrCtx>> drr_ctx_list(MakeDrrCtxList());
    return drr_ctx_list;
  }

  adt::Result<adt::List<DrrCtx>> MakeDrrCtxList() {
    adt::List<DrrCtx> ret{};
    auto Collect = [&](const auto& drr_ctx) -> adt::Result<adt::Ok> {
      ret->emplace_back(drr_ctx);
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(VisitEachDrrCtxByClassicDrrPassRegistryItems(Collect));
    return ret;
  }

  adt::Result<adt::Ok> PostProcess(
      adt::Result<std::optional<GraphMatchCtx>> (*Match)(const DrrCtx&,
                                                         pir::Operation* op),
      const DrrCtx& drr_ctx,
      pir::Operation* op,
      const GraphMatchCtx& match_ctx,
      const std::function<adt::Result<CodeGenResult>(const std::string&)>&
          CodeGenResult4FusedOpName) override {
    // Do nothing.
    return adt::Ok{};
  }

 private:
  template <typename YieldT>
  adt::Result<adt::Ok> VisitEachDrrCtxByClassicDrrPassRegistryItems(
      const YieldT& Yield) {
    ADT_LET_CONST_REF(registry, ApRegistryHelper{}.SingletonRegistry());
    const auto& classic_drr_pass_registry_items =
        registry->classic_drr_pass_registry_items;
    for (const auto& [classic_drr_pass_name, nice2classic_drr_pass_items] :
         classic_drr_pass_registry_items) {
      std::optional<DrrCtx> opt_drr_ctx;
      for (const auto& [nice, classic_drr_pass_items] :
           nice2classic_drr_pass_items) {
        if (opt_drr_ctx.has_value()) {
          break;
        }
        for (const auto& classic_drr_pass_item : classic_drr_pass_items) {
          const auto& drr_ctx = GetDrrCtx(classic_drr_pass_item);
          if (drr_ctx.HasOkValue()) {
            ADT_RETURN_IF_ERR(Yield(drr_ctx.GetOkValue()));
            opt_drr_ctx = drr_ctx.GetOkValue();
            break;
          } else {
            LOG(ERROR) << "\nTraceback (most recent call last):\n"
                       << drr_ctx.GetError().CallStackToString() << "\n"
                       << drr_ctx.GetError().class_name()
                       << ": classic_drr_pass_name: " << classic_drr_pass_name
                       << " nice: " << nice
                       << " msg: " << drr_ctx.GetError().msg();
          }
        }
      }
    }
    return adt::Ok{};
  }

  adt::Result<DrrCtx> GetDrrCtx(
      const ap::registry::ClassicDrrPassRegistryItem& classic_drr_pass_item) {
    ADT_LET_CONST_REF(
        drr_ctx,
        ApDrrHelper{circlable_ref_list_}.Interpret(classic_drr_pass_item->cls));
    if (!drr_ctx->pass_name.has_value()) {
      drr_ctx.shared_ptr()->pass_name =
          classic_drr_pass_item->classic_drr_pass_name;
    }
    return drr_ctx;
  }
};

class TagAccessTopoDrrCtxProvider : public DrrCtxProvider {
  std::weak_ptr<ap::memory::CirclableRefListBase> circlable_ref_list_;
  std::string pass_tag_name_;

 public:
  explicit TagAccessTopoDrrCtxProvider(
      const std::weak_ptr<ap::memory::CirclableRefListBase>& circlable_ref_list,
      const std::string& pass_tag_name)
      : circlable_ref_list_(circlable_ref_list),
        pass_tag_name_(pass_tag_name) {}

  adt::Result<adt::List<DrrCtx>> GetDrrCtxList() override {
    adt::Result<adt::List<DrrCtx>> drr_ctx_list(MakeDrrCtxList(pass_tag_name_));
    return drr_ctx_list;
  }

  adt::Result<adt::List<DrrCtx>> MakeDrrCtxList(
      const std::string& pass_tag_name) {
    adt::List<DrrCtx> ret{};
    auto Collect = [&](const auto& drr_ctx) -> adt::Result<adt::Ok> {
      ret->emplace_back(drr_ctx);
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(VisitEachDrrCtxByAccessTopoDrrPassRegistryItems(
        pass_tag_name, Collect));
    return ret;
  }

  adt::Result<adt::Ok> PostProcess(
      adt::Result<std::optional<GraphMatchCtx>> (*Match)(const DrrCtx&,
                                                         pir::Operation* op),
      const DrrCtx& drr_ctx,
      pir::Operation* op,
      const GraphMatchCtx& match_ctx,
      const std::function<adt::Result<CodeGenResult>(const std::string&)>&
          CodeGenResult4FusedOpName) override {
    // Do nothing.
    return adt::Ok{};
  }

 private:
  template <typename YieldT>
  adt::Result<adt::Ok> VisitEachDrrCtxByAccessTopoDrrPassRegistryItems(
      const std::string& pass_tag_name, const YieldT& Yield) {
    ADT_LET_CONST_REF(registry, ApRegistryHelper{}.SingletonRegistry());
    const auto& access_topo_drr_pass_registry_items =
        registry->access_topo_drr_pass_registry_items;
    for (const auto& [access_topo_drr_pass_name,
                      nice2access_topo_drr_pass_items] :
         access_topo_drr_pass_registry_items) {
      std::optional<DrrCtx> opt_drr_ctx;
      for (const auto& [nice, access_topo_drr_pass_items] :
           nice2access_topo_drr_pass_items) {
        if (opt_drr_ctx.has_value()) {
          break;
        }
        for (const auto& access_topo_drr_pass_item :
             access_topo_drr_pass_items) {
          if (pass_tag_name != access_topo_drr_pass_item->pass_tag_name) {
            continue;
          }
          const auto& drr_ctx = GetDrrCtx(access_topo_drr_pass_item);
          if (drr_ctx.HasOkValue()) {
            ADT_RETURN_IF_ERR(Yield(drr_ctx.GetOkValue()));
            opt_drr_ctx = drr_ctx.GetOkValue();
            break;
          } else {
            LOG(ERROR) << "\nTraceback (most recent call last):\n"
                       << drr_ctx.GetError().CallStackToString() << "\n"
                       << drr_ctx.GetError().class_name()
                       << ": access_topo_drr_pass_name: "
                       << access_topo_drr_pass_name << " nice: " << nice
                       << " msg: " << drr_ctx.GetError().msg();
          }
        }
      }
    }
    return adt::Ok{};
  }

  adt::Result<DrrCtx> GetDrrCtx(
      const ap::registry::AccessTopoDrrPassRegistryItem&
          access_topo_drr_pass_item) {
    ADT_LET_CONST_REF(drr_ctx,
                      ApDrrHelper{circlable_ref_list_}.Interpret(
                          access_topo_drr_pass_item->cls));
    if (!drr_ctx->pass_name.has_value()) {
      drr_ctx.shared_ptr()->pass_name =
          access_topo_drr_pass_item->access_topo_drr_pass_name;
    }
    return drr_ctx;
  }
};

class CustomAccessTopoDrrCtxProvider : public DrrCtxProvider {
  std::weak_ptr<ap::memory::CirclableRefListBase> circlable_ref_list_;
  ap::axpr::Value drr_pass_obj_;
  ap::axpr::Value mut_matched_pattern_as_programs_;
  std::size_t seq_no_;

 public:
  explicit CustomAccessTopoDrrCtxProvider(
      const std::weak_ptr<ap::memory::CirclableRefListBase>& circlable_ref_list,
      const ap::axpr::Value& drr_pass_obj,
      const ap::axpr::Value& mut_matched_pattern_as_programs)
      : circlable_ref_list_(circlable_ref_list),
        drr_pass_obj_(drr_pass_obj),
        mut_matched_pattern_as_programs_(mut_matched_pattern_as_programs),
        seq_no_(0) {}

  adt::Result<adt::List<DrrCtx>> GetDrrCtxList() override {
    adt::Result<adt::List<DrrCtx>> drr_ctx_list(MakeDrrCtxList());
    return drr_ctx_list;
  }

  adt::Result<adt::List<DrrCtx>> MakeDrrCtxList() {
    adt::List<DrrCtx> ret{};
    auto Collect = [&](const auto& drr_ctx) -> adt::Result<adt::Ok> {
      ret->emplace_back(drr_ctx);
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(VisitEachCreatedDrrCtx(Collect));
    return ret;
  }

  adt::Result<adt::Ok> PostProcess(
      adt::Result<std::optional<GraphMatchCtx>> (*Match)(const DrrCtx&,
                                                         pir::Operation* op),
      const DrrCtx& drr_ctx,
      pir::Operation* op,
      const GraphMatchCtx& match_ctx,
      const std::function<adt::Result<CodeGenResult>(const std::string&)>&
          CodeGenResult4FusedOpName) override {
    if (mut_matched_pattern_as_programs_.template CastableTo<adt::Nothing>()) {
      return adt::Ok{};
    }
    ADT_LET_CONST_REF(
        mut_lst,
        mut_matched_pattern_as_programs_
            .template CastTo<ap::axpr::MutableList<ap::axpr::Value>>());
    ADT_CHECK(drr_ctx->source_pattern_ctx.has_value());
    ADT_LET_CONST_REF(program,
                      CopyMatchedPatternToProgram(
                          drr_ctx->source_pattern_ctx.value(), match_ctx));
    ADT_LET_CONST_REF(mut_list_ptr, mut_lst.Mut());
    mut_list_ptr->emplace_back(ap::paddle::GetPirProgramClass().New(program));
    return adt::Ok{};
  }

 private:
  adt::Result<ap::paddle::Program> CopyMatchedPatternToProgram(
      const ap::drr::SourcePatternCtx& source_pattern_ctx,
      const GraphMatchCtx& match_ctx) {
    pir::IrContext* ctx = ::pir::IrContext::Instance();
    auto new_program = std::make_shared<::pir::Program>(ctx);
    auto clone_options = ::pir::CloneOptions::All();
    pir::IrMapping ir_mapping;
    pir::Builder builder(ctx, new_program->block());
    auto DoEachInput = [&](const auto& name,
                           pir::Value value) -> adt::Result<adt::Ok> {
      ADT_CHECK(value.type().isa<pir::DenseTensorType>());
      const auto& type = value.type().dyn_cast<pir::DenseTensorType>();
      const auto& dims = ::common::vectorize(type.dims());
      auto phi_type = ::paddle::dialect::TransToPhiDataType(type.dtype());
      auto op = builder.Build<::paddle::dialect::DataOp>(
          name, dims, phi_type, phi::Place());
      ir_mapping.Add(value, op->result(0));
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(
        VisitMatchedInput(source_pattern_ctx, match_ctx, DoEachInput));
    std::optional<pir::Program*> old_program{};
    auto DoEachOp = [&](pir::Operation* op) -> adt::Result<adt::Ok> {
      if (old_program.has_value()) {
        ADT_CHECK(old_program.value() == op->GetParentProgram());
      } else {
        old_program = op->GetParentProgram();
      }
      auto* new_op = op->Clone(ir_mapping, clone_options);
      new_program->block()->push_back(new_op);
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(VisitMatchedOp(source_pattern_ctx, match_ctx, DoEachOp));
    if (old_program.has_value()) {
      ADT_RETURN_IF_ERR(CloneSymbolicShapes(
          new_program.get(), old_program.value(), ir_mapping));
    }
    return ap::paddle::Program{new_program};
  }

  adt::Result<adt::Ok> CloneSymbolicShapes(pir::Program* new_program,
                                           pir::Program* old_program,
                                           const pir::IrMapping& ir_mapping) {
    auto* new_shape_analysis =
        &::pir::ShapeAnalysisManager::Instance().Get(new_program);
    auto* old_shape_analysis =
        &::pir::ShapeAnalysisManager::Instance().Get(old_program);
    for (const auto& [old_value, new_value] : ir_mapping.GetMap<pir::Value>()) {
      new_shape_analysis->SetShapeOrDataForValue(
          new_value, old_shape_analysis->GetShapeOrDataForValue(old_value));
    }
    return adt::Ok{};
  }

  template <typename YieldT>
  adt::Result<adt::Ok> VisitMatchedOp(
      const ap::drr::SourcePatternCtx& source_pattern_ctx,
      const GraphMatchCtx& match_ctx,
      const YieldT& Yield) {
    using Ok = adt::Result<adt::Ok>;
    auto DoEachOp = [&](const auto& drr_graph_node) -> Ok {
      ADT_LET_CONST_REF(drr_node, drr_graph_node.Get());
      const auto& drr_op = GetDrrOp(drr_node);
      if (!drr_op.has_value()) return adt::Ok{};
      ADT_LET_CONST_REF(pir_node,
                        match_ctx->GetSoleBigGraphNode(drr_graph_node));
      const auto& pir_op = GetPirOp(pir_node);
      if (pir_op.has_value()) {
        ADT_RETURN_IF_ERR(Yield(pir_op.value()));
      }
      return adt::Ok{};
    };
    return VisitEachGraphNode(source_pattern_ctx, DoEachOp);
  }

  template <typename YieldT>
  adt::Result<adt::Ok> VisitEachGraphNode(
      const ap::drr::SourcePatternCtx& source_pattern_ctx,
      const YieldT& Yield) {
    std::list<DrrGraphNode> sources;
    for (const auto& drr_node : source_pattern_ctx->node_arena->nodes()) {
      const auto& drr_graph_node = drr_node.node();
      ADT_LET_CONST_REF(upstreams, drr_graph_node.UpstreamNodes());
      if (upstreams.size() == 0) {
        sources.push_back(drr_graph_node);
      }
    }
    using Ok = adt::Result<adt::Ok>;
    ap::drr::DefaultDrrGraphDescriptor graph{};
    auto VisitPrev = [&](const DrrGraphNode& node, const auto& Yield) -> Ok {
      return graph.VisitUpstreamNodes(node, Yield);
    };
    auto VisitNext = [&](const DrrGraphNode& node, const auto& Yield) -> Ok {
      return graph.VisitDownstreamNodes(node, Yield);
    };
    ap::adt::TopoWalker<DrrGraphNode> walker{VisitPrev, VisitNext};
    ADT_RETURN_IF_ERR(walker(sources.begin(), sources.end(), Yield));
    return adt::Ok{};
  }

  std::optional<DrrIrOp> GetDrrOp(const DrrNode& drr_node) const {
    return drr_node.Match(
        [&](const DrrNativeIrOp& ir_op) -> std::optional<DrrIrOp> {
          return DrrIrOp{ir_op};
        },
        [&](const DrrPackedIrOp& ir_op) -> std::optional<DrrIrOp> {
          return DrrIrOp{ir_op};
        },
        [&](const auto&) -> std::optional<DrrIrOp> { return std::nullopt; });
  }

  std::optional<pir::Operation*> GetPirOp(const PirNode& pir_node) const {
    return pir_node.Match(
        [&](const ap::paddle::NativeIrOp& ir_op)
            -> std::optional<pir::Operation*> { return ir_op.op; },
        [&](const ap::paddle::PackedIrOp& ir_op)
            -> std::optional<pir::Operation*> { return ir_op.fusion_op; },
        [&](const auto&) -> std::optional<pir::Operation*> {
          return std::nullopt;
        });
  }

  template <typename YieldT>
  adt::Result<adt::Ok> VisitMatchedInput(
      const ap::drr::SourcePatternCtx& source_pattern_ctx,
      const GraphMatchCtx& match_ctx,
      const YieldT& Yield) {
    for (const auto& drr_node : source_pattern_ctx->node_arena->nodes()) {
      const auto& drr_graph_node = drr_node.node();
      ADT_LET_CONST_REF(upstreams, drr_graph_node.UpstreamNodes());
      if (upstreams.size() > 0) {
        continue;
      }
      ADT_RETURN_IF_ERR(drr_node.Match(
          [&](const DrrNativeIrValue& impl) -> adt::Result<adt::Ok> {
            return VisitMatchedNativeIrValueInput(impl, match_ctx, Yield);
          },
          [&](const DrrPackedIrValue& impl) -> adt::Result<adt::Ok> {
            return VisitMatchedPackedIrValueInput(impl, match_ctx, Yield);
          },
          [&](const DrrNativeIrOp&) -> adt::Result<adt::Ok> {
            // Do nothing.
            return adt::Ok{};
          },
          [&](const DrrPackedIrOp&) -> adt::Result<adt::Ok> {
            // Do nothing.
            return adt::Ok{};
          },
          [&](const DrrOptPackedIrOp&) -> adt::Result<adt::Ok> {
            // Do nothing.
            return adt::Ok{};
          },
          [&](const auto&) -> adt::Result<adt::Ok> {
            return adt::errors::NotImplementedError{
                "VisitMatchedInput() failed"};
          }));
    }
    return adt::Ok{};
  }

  template <typename YieldT>
  adt::Result<adt::Ok> VisitMatchedNativeIrValueInput(
      const DrrNativeIrValue& drr_value,
      const GraphMatchCtx& match_ctx,
      const YieldT& Yield) {
    auto DoEach = [&](const PirNode& pir_node) -> adt::Result<adt::Ok> {
      ADT_LET_CONST_REF(native_ir_value,
                        pir_node.template TryGet<ap::paddle::NativeIrValue>());
      return Yield(drr_value->name, native_ir_value.value);
    };
    return match_ctx->VisitBigGraphIrValueNode(drr_value->node, DoEach);
  }

  template <typename YieldT>
  adt::Result<adt::Ok> VisitMatchedPackedIrValueInput(
      const DrrPackedIrValue& drr_value,
      const GraphMatchCtx& match_ctx,
      const YieldT& Yield) {
    int i = 0;
    auto DoEach = [&](const PirNode& pir_node) -> adt::Result<adt::Ok> {
      ADT_LET_CONST_REF(native_ir_value,
                        pir_node.template TryGet<ap::paddle::NativeIrValue>());
      const auto& name = drr_value->name + "[" + std::to_string(i++) + "]";
      return Yield(drr_value->name, native_ir_value.value);
    };
    return match_ctx->VisitBigGraphIrValueNode(drr_value->node, DoEach);
  }

  template <typename YieldT>
  adt::Result<adt::Ok> VisitEachCreatedDrrCtx(const YieldT& Yield) {
    using AList = ap::axpr::AbstractList<ap::axpr::Value>;
    if (AList::CastableFrom(drr_pass_obj_)) {
      auto DoEach =
          [&](const auto& drr_pass_obj) -> adt::Result<adt::LoopCtrl> {
        ADT_LET_CONST_REF(drr_ctx, GetDrrCtx(drr_pass_obj));
        ADT_RETURN_IF_ERR(Yield(drr_ctx));
        return adt::Continue{};
      };
      ADT_LET_CONST_REF(lst, AList::CastFrom(drr_pass_obj_));
      ADT_RETURN_IF_ERR(lst.Visit(DoEach));
    } else {
      ADT_LET_CONST_REF(drr_ctx, GetDrrCtx(drr_pass_obj_));
      ADT_RETURN_IF_ERR(Yield(drr_ctx));
    }
    return adt::Ok{};
  }

  adt::Result<DrrCtx> GetDrrCtx(const ap::axpr::Value& drr_pass_obj) {
    ApDrrHelper helper{circlable_ref_list_};
    ADT_LET_CONST_REF(drr_ctx, helper.CreateDrrCtxByDrrPassObj(drr_pass_obj));
    if (!drr_ctx->pass_name.has_value()) {
      drr_ctx.shared_ptr()->pass_name =
          std::string("tmp_access_drr_pass_") + std::to_string(seq_no_++);
    }
    return drr_ctx;
  }
};

adt::Result<ap::registry::Registry> TryGetRegistrySingleton() {
  ap::paddle::ForceLinkPir();
  ap::paddle::ForceLinkIrTools();
  ADT_LET_CONST_REF(registry, ApRegistryHelper{}.SingletonRegistry());
  return registry;
}

std::optional<ap::registry::Registry> GetRegistrySingleton() {
  const auto& registry = TryGetRegistrySingleton();
  if (registry.HasOkValue()) {
    return registry.GetOkValue();
  } else {
    LOG(ERROR) << "\nTraceback (most recent call last):\n"
               << registry.GetError().CallStackToString() << "\n"
               << registry.GetError().class_name() << ": "
               << registry.GetError().msg();
    return std::nullopt;
  }
}

}  // namespace

std::optional<std::unique_ptr<::pir::Pass>> CreateApGenericAbstractDrrPass(
    const std::weak_ptr<ap::memory::CirclableRefListBase>& circlable_ref_list) {
  if (!GetRegistrySingleton().has_value()) {
    return std::nullopt;
  }
  auto drr_ctx_provider =
      std::make_shared<AbstractDrrCtxProvider>(circlable_ref_list);
  const auto& drr_ctx_list = drr_ctx_provider->GetDrrCtxList();
  if (drr_ctx_list.HasError()) {
    LOG(ERROR) << "\nTraceback (most recent call last):\n"
               << drr_ctx_list.GetError().CallStackToString() << "\n"
               << drr_ctx_list.GetError().class_name() << ": "
               << drr_ctx_list.GetError().msg();
    return std::nullopt;
  }
  if (drr_ctx_list.GetOkValue()->empty()) {
    return std::nullopt;
  }
  std::unique_ptr<::pir::Pass> pass =
      std::make_unique<ApGenericDrrPass>(drr_ctx_provider,
                                         /*name=*/"abstract",
                                         /*steps_limit=*/std::nullopt);
  return std::move(pass);
}

std::optional<std::unique_ptr<::pir::Pass>> CreateApGenericClassicDrrPass(
    const std::weak_ptr<ap::memory::CirclableRefListBase>& circlable_ref_list) {
  if (!GetRegistrySingleton().has_value()) {
    return std::nullopt;
  }
  auto drr_ctx_provider =
      std::make_shared<ClassicDrrCtxProvider>(circlable_ref_list);
  const auto& drr_ctx_list = drr_ctx_provider->GetDrrCtxList();
  if (drr_ctx_list.HasError()) {
    LOG(ERROR) << "\nTraceback (most recent call last):\n"
               << drr_ctx_list.GetError().CallStackToString() << "\n"
               << drr_ctx_list.GetError().class_name() << ": "
               << drr_ctx_list.GetError().msg();
    return std::nullopt;
  }
  if (drr_ctx_list.GetOkValue()->empty()) {
    return std::nullopt;
  }
  std::unique_ptr<::pir::Pass> pass =
      std::make_unique<ApGenericDrrPass>(drr_ctx_provider,
                                         /*name=*/"classic",
                                         /*steps_limit=*/std::nullopt);
  return std::move(pass);
}

std::optional<std::unique_ptr<::pir::Pass>> CreateAccessTopoDrrPass(
    const std::weak_ptr<ap::memory::CirclableRefListBase>& circlable_ref_list,
    const std::string& drr_pass_tag,
    std::optional<int64_t> steps_limit) {
  if (!GetRegistrySingleton().has_value()) {
    return std::nullopt;
  }
  auto drr_ctx_provider = std::make_shared<TagAccessTopoDrrCtxProvider>(
      circlable_ref_list, drr_pass_tag);
  const auto& drr_ctx_list = drr_ctx_provider->GetDrrCtxList();
  if (drr_ctx_list.HasError()) {
    LOG(ERROR) << "\nTraceback (most recent call last):\n"
               << drr_ctx_list.GetError().CallStackToString() << "\n"
               << drr_ctx_list.GetError().class_name() << ": "
               << drr_ctx_list.GetError().msg();
    return std::nullopt;
  }
  if (drr_ctx_list.GetOkValue()->empty()) {
    return std::nullopt;
  }
  std::unique_ptr<::pir::Pass> pass =
      std::make_unique<ApGenericDrrPass>(drr_ctx_provider,
                                         /*name=*/"tag_access_topo",
                                         /*steps_limit=*/steps_limit);
  return std::move(pass);
}

std::optional<std::unique_ptr<::pir::Pass>> CreateCustomAccessTopoDrrPass(
    const std::weak_ptr<ap::memory::CirclableRefListBase>& circlable_ref_list,
    const ap::axpr::Value& drr_pass_obj,
    std::optional<int64_t> steps_limit,
    const ap::axpr::Value& mut_matched_pattern_as_programs) {
  auto drr_ctx_provider = std::make_shared<CustomAccessTopoDrrCtxProvider>(
      circlable_ref_list, drr_pass_obj, mut_matched_pattern_as_programs);
  const auto& drr_ctx_list = drr_ctx_provider->GetDrrCtxList();
  if (drr_ctx_list.HasError()) {
    LOG(ERROR) << "\nTraceback (most recent call last):\n"
               << drr_ctx_list.GetError().CallStackToString() << "\n"
               << drr_ctx_list.GetError().class_name() << ": "
               << drr_ctx_list.GetError().msg();
    return std::nullopt;
  }
  if (drr_ctx_list.GetOkValue()->empty()) {
    return std::nullopt;
  }
  std::unique_ptr<::pir::Pass> pass =
      std::make_unique<ApGenericDrrPass>(drr_ctx_provider,
                                         /*name=*/"custom_access_topo",
                                         /*steps_limit=*/steps_limit);
  return std::move(pass);
}

}  // namespace cinn::dialect::ir
