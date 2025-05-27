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

#include "paddle/ap/include/reified_drr/reified_src_ptn_axpr_maker.h"
#include "paddle/ap/include/drr/node.h"
#include "paddle/ap/include/drr/op_tensor_pattern_ctx_helper.h"

namespace ap::reified_drr {

namespace {

using SrcPtnCtxOpImpl = std::variant<drr::NativeIrOp<drr::Node>,
                                     drr::PackedIrOp<drr::Node>,
                                     drr::OptPackedIrOp<drr::Node>>;

struct SrcPtnCtxOp : public SrcPtnCtxOpImpl {
  using SrcPtnCtxOpImpl::SrcPtnCtxOpImpl;
  ADT_DEFINE_VARIANT_METHODS(SrcPtnCtxOpImpl);

  const std::string& op_unique_name() const {
    using RetT = const std::string&;
    return Match([&](const auto& impl) -> RetT { return impl->name; });
  }

  static std::optional<SrcPtnCtxOp> CastFromDrrNode(const drr::Node& drr_node) {
    using RetT = std::optional<SrcPtnCtxOp>;
    return drr_node.Match(
        [&](const drr::NativeIrOp<drr::Node>& impl) -> RetT { return impl; },
        [&](const drr::PackedIrOp<drr::Node>& impl) -> RetT { return impl; },
        [&](const drr::OptPackedIrOp<drr::Node>& impl) -> RetT { return impl; },
        [&](const auto&) -> RetT { return std::nullopt; });
  }

  adt::Result<std::vector<std::optional<std::string>>> GetInputValueNames()
      const {
    std::vector<std::optional<std::string>> ret;
    ADT_LET_CONST_REF(reserved_size, num_inputs());
    ret.reserve(reserved_size);
    auto CollectValueName =
        [&](const auto& op_operand) -> adt::Result<adt::Ok> {
      ADT_LET_CONST_REF(upstreams, op_operand.UpstreamNodes());
      if (upstreams.size() == 0) {
        ret.emplace_back(std::nullopt);
      } else {
        ADT_LET_CONST_REF(input_node, upstreams.Sole());
        ADT_LET_CONST_REF(input, input_node.Get());
        const auto& ir_value = drr::IrValue::OptCastFrom(input);
        ADT_CHECK(ir_value.has_value());
        ret.emplace_back(ir_value.value().name());
      }
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(VisitUpstream(CollectValueName));
    return ret;
  }

  adt::Result<std::vector<std::string>> GetOutputValueNames() const {
    std::vector<std::string> ret;
    ADT_LET_CONST_REF(reserved_size, num_outputs());
    ret.reserve(reserved_size);
    auto CollectValueName = [&](const auto& op_result) -> adt::Result<adt::Ok> {
      ADT_LET_CONST_REF(downstreams, op_result.DownstreamNodes());
      ADT_LET_CONST_REF(output_node, downstreams.Sole());
      ADT_LET_CONST_REF(output, output_node.Get());
      const auto& ir_value = drr::IrValue::OptCastFrom(output);
      ADT_CHECK(ir_value.has_value());
      ret.emplace_back(ir_value.value().name());
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(VisitDownstream(CollectValueName));
    return ret;
  }

  template <typename DoEachT>
  adt::Result<adt::Ok> VisitUpstream(const DoEachT& DoEach) const {
    return Match([&](const auto& op) -> adt::Result<adt::Ok> {
      ADT_LET_CONST_REF(upstreams, op->node.UpstreamNodes());
      ADT_RETURN_IF_ERR(upstreams.VisitNodes(DoEach));
      return adt::Ok{};
    });
  }

  template <typename DoEachT>
  adt::Result<adt::Ok> VisitDownstream(const DoEachT& DoEach) const {
    return Match([&](const auto& op) -> adt::Result<adt::Ok> {
      ADT_LET_CONST_REF(downstreams, op->node.DownstreamNodes());
      ADT_RETURN_IF_ERR(downstreams.VisitNodes(DoEach));
      return adt::Ok{};
    });
  }

  adt::Result<std::size_t> num_inputs() const {
    return Match([&](const auto& op) -> adt::Result<std::size_t> {
      ADT_LET_CONST_REF(upstreams, op->node.UpstreamNodes());
      return upstreams.size();
    });
  }

  adt::Result<std::size_t> num_outputs() const {
    return Match([&](const auto& op) -> adt::Result<std::size_t> {
      ADT_LET_CONST_REF(downstreams, op->node.DownstreamNodes());
      return downstreams.size();
    });
  }
};

adt::Result<adt::Ok> GenAnfExprForOpImpl(
    axpr::LetVar* op_pattern_ctx,
    DrrNodeAttrToAnfExprHelper* anf_expr_helper,
    MatchedSrcPtnCtxHelper* matched_src_ptn_ctx_helper,
    const std::string& op_unique_name,
    const drr::NativeIrOp<drr::Node>& ir_op) {
  const auto& op =
      op_pattern_ctx->Attr("ap_native_op").Call(ir_op->op_declare->op_name);
  op_pattern_ctx->SetAttr(op_unique_name, op);
  using Ok = adt::Result<adt::Ok>;
  auto GenSetAttr = [&](const std::string& attr_name,
                        const axpr::Value& attr_val) -> Ok {
    ADT_LET_CONST_REF(
        attr_anf_expr,
        anf_expr_helper->ConvertAttrToAnfExpr(op_pattern_ctx->ctx(), attr_val));
    op_pattern_ctx->Attr(op_unique_name).SetAttr(attr_name, attr_anf_expr);
    return adt::Ok{};
  };
  ADT_RETURN_IF_ERR(
      matched_src_ptn_ctx_helper->VisitNativeIrOpAttr(ir_op, GenSetAttr));
  return adt::Ok{};
}

adt::Result<axpr::AnfExpr> GenInnerSrcPtnLambda(
    DrrNodeAttrToAnfExprHelper* anf_expr_helper,
    MatchedSrcPtnCtxHelper* matched_src_ptn_ctx_helper,
    const drr::PackedIrOp<drr::Node>& ir_op) {
  ADT_LET_CONST_REF(
      inner_matched_src_ptn_ctx_helper,
      matched_src_ptn_ctx_helper->MakeInnerMatchedSrcPtnCtxHelper(ir_op));
  axpr::LambdaExprBuilder lmd{};
  ReifiedSrcPtnAxprMaker maker{anf_expr_helper,
                               inner_matched_src_ptn_ctx_helper.get()};
  auto GetBody = [&](auto& ctx) -> adt::Result<axpr::AnfExpr> {
    auto* op_pattern_ctx = &ctx.Var("o");
    auto* tensor_pattern_ctx = &ctx.Var("t");
    ADT_RETURN_IF_ERR(maker.GenAnfExprForSrcPtnCtxOps(op_pattern_ctx));
    ADT_RETURN_IF_ERR(maker.GenAnfExprForSrcPtnCtxValues(tensor_pattern_ctx));
    ADT_RETURN_IF_ERR(maker.GenAnfExprForSrcPtnCtxOpValueConnections(
        op_pattern_ctx, tensor_pattern_ctx));
    return ctx.None();
  };
  ADT_LET_CONST_REF(ret, lmd.TryLambda({"o", "t"}, GetBody));
  return ret;
}

adt::Result<adt::Ok> GenAnfExprForOpImpl(
    axpr::LetVar* op_pattern_ctx,
    DrrNodeAttrToAnfExprHelper* anf_expr_helper,
    MatchedSrcPtnCtxHelper* matched_src_ptn_ctx_helper,
    const std::string& op_unique_name,
    const drr::PackedIrOp<drr::Node>& ir_op) {
  ADT_LET_CONST_REF(
      inner_src_ptn_lambda,
      GenInnerSrcPtnLambda(anf_expr_helper, matched_src_ptn_ctx_helper, ir_op));
  auto* ctx = op_pattern_ctx->ctx();
  const std::string& lambda_name = ctx->NewTmpVarName();
  ctx->Var(lambda_name) = inner_src_ptn_lambda;
  const auto& inner_src_ptn_func = ctx->Var(lambda_name).Attr("__function__");
  const auto& op =
      op_pattern_ctx->Attr("ap_trivial_fusion_op").Call(inner_src_ptn_func);
  op_pattern_ctx->SetAttr(op_unique_name, op);
  return adt::Ok{};
}

adt::Result<adt::Ok> GenAnfExprForOpImpl(
    axpr::LetVar* op_pattern_ctx,
    DrrNodeAttrToAnfExprHelper* anf_expr_helper,
    MatchedSrcPtnCtxHelper* matched_src_ptn_ctx_helper,
    const std::string& op_unique_name,
    const drr::OptPackedIrOp<drr::Node>& ir_op) {
  return adt::errors::NotImplementedError{
      "GenAnfExprForOpImpl(OptPackedIrOp) not implemented"};
}

}  // namespace

namespace {

template <typename DoEachT>
adt::Result<adt::Ok> VisitEachSrcPtnCtxOp(
    const drr::SourcePatternCtx& src_ptn_ctx, const DoEachT& DoEach) {
  for (const auto& node : src_ptn_ctx->node_arena->nodes()) {
    const auto& src_ptn_ctx_op = SrcPtnCtxOp::CastFromDrrNode(node);
    if (src_ptn_ctx_op.has_value()) {
      ADT_RETURN_IF_ERR(DoEach(src_ptn_ctx_op.value()));
    }
  }
  return adt::Ok{};
}

}  // namespace

adt::Result<adt::Ok> ReifiedSrcPtnAxprMaker::GenAnfExprForSrcPtnCtxOps(
    axpr::LetVar* op_pattern_ctx) {
  using Ok = adt::Result<adt::Ok>;
  auto GenAnfExprForOp = [&](const SrcPtnCtxOp& op) -> Ok {
    return op.Match([&](const auto& impl) -> Ok {
      return GenAnfExprForOpImpl(op_pattern_ctx,
                                 anf_expr_helper_,
                                 matched_src_ptn_ctx_helper_,
                                 op.op_unique_name(),
                                 impl);
    });
  };
  const auto& src_ptn_ctx = matched_src_ptn_ctx_helper_->src_ptn_ctx();
  ADT_RETURN_IF_ERR(VisitEachSrcPtnCtxOp(src_ptn_ctx, GenAnfExprForOp));
  return adt::Ok{};
}

namespace {

template <typename DoEachT>
adt::Result<adt::Ok> VisitEachSrcPtnCtxValue(
    const drr::SourcePatternCtx& src_ptn_ctx, const DoEachT& DoEach) {
  for (const auto& node : src_ptn_ctx->node_arena->nodes()) {
    if (node.template Has<drr::NativeIrValue<drr::Node>>()) {
      ADT_RETURN_IF_ERR(
          DoEach(node.template Get<drr::NativeIrValue<drr::Node>>()));
    }
  }
  return adt::Ok{};
}

}  // namespace

adt::Result<adt::Ok> ReifiedSrcPtnAxprMaker::GenAnfExprForSrcPtnCtxValues(
    axpr::LetVar* tensor_pattern_ctx) {
  using Ok = adt::Result<adt::Ok>;
  auto GenAnfExprForValue =
      [&](const drr::NativeIrValue<drr::Node>& drr_value) -> Ok {
    ADT_LET_CONST_REF(
        type, matched_src_ptn_ctx_helper_->GetNativeIrValueType(drr_value));
    ADT_LET_CONST_REF(type_anf_expr,
                      anf_expr_helper_->ConvertTypeToAnfExpr(
                          tensor_pattern_ctx->ctx(), type));
    tensor_pattern_ctx->Attr(drr_value->name).SetAttr("type", type_anf_expr);
    return adt::Ok{};
  };
  const auto& src_ptn_ctx = matched_src_ptn_ctx_helper_->src_ptn_ctx();
  ADT_RETURN_IF_ERR(VisitEachSrcPtnCtxValue(src_ptn_ctx, GenAnfExprForValue));
  return adt::Ok{};
}

namespace {

template <typename DoEachT>
adt::Result<adt::Ok> VisitEachSrcPtnCtxOpValueConnection(
    const drr::SourcePatternCtx& src_ptn_ctx, const DoEachT& DoEach) {
  auto DoEachOp = [&](const SrcPtnCtxOp& op) -> adt::Result<adt::Ok> {
    ADT_LET_CONST_REF(input_value_names, op.GetInputValueNames());
    ADT_LET_CONST_REF(output_value_names, op.GetOutputValueNames());
    ADT_RETURN_IF_ERR(
        DoEach(op.op_unique_name(), input_value_names, output_value_names));
    return adt::Ok{};
  };
  ADT_RETURN_IF_ERR(VisitEachSrcPtnCtxOp(src_ptn_ctx, DoEachOp));
  return adt::Ok{};
}

}  // namespace

adt::Result<adt::Ok>
ReifiedSrcPtnAxprMaker::GenAnfExprForSrcPtnCtxOpValueConnections(
    axpr::LetVar* op_pattern_ctx, axpr::LetVar* tensor_pattern_ctx) {
  ADT_CHECK(op_pattern_ctx->ctx() == tensor_pattern_ctx->ctx());
  auto* ctx = op_pattern_ctx->ctx();
  using Ok = adt::Result<adt::Ok>;
  const auto& src_ptn_ctx = matched_src_ptn_ctx_helper_->src_ptn_ctx();
  auto GetDrrIrValueAxpr =
      [&](const auto& tensor_name) -> adt::Result<axpr::AnfExpr> {
    ADT_LET_CONST_REF(ir_value,
                      drr::OpTensorPatternCtxHelper{}.GetIrValueByUid(
                          src_ptn_ctx->tensor_pattern_ctx, tensor_name));
    using RetT = adt::Result<axpr::AnfExpr>;
    return ir_value.Match(
        [&](const drr::NativeIrValue<drr::Node>&) -> RetT {
          return tensor_pattern_ctx->Attr(tensor_name);
        },
        [&](const drr::PackedIrValue<drr::Node>&) -> RetT {
          auto* ctx = tensor_pattern_ctx->ctx();
          return ctx->Var(axpr::kBuiltinStarred())
              .Call(tensor_pattern_ctx->Attr(tensor_name));
        });
  };
  auto BuildConnection =
      [&](const std::string& op_unique_name,
          const std::vector<std::optional<std::string>>& in_names,
          const std::vector<std::string>& out_names) -> Ok {
    std::vector<axpr::AnfExpr> in_anf_exprs;
    in_anf_exprs.reserve(in_names.size());
    for (const auto& opt_name : in_names) {
      if (!opt_name.has_value()) {
        in_anf_exprs.emplace_back(ctx->None());
      } else {
        ADT_LET_CONST_REF(anf_expr, GetDrrIrValueAxpr(opt_name.value()));
        in_anf_exprs.emplace_back(anf_expr);
      }
    }
    std::vector<axpr::AnfExpr> out_anf_exprs;
    for (const auto& name : out_names) {
      ADT_LET_CONST_REF(anf_expr, GetDrrIrValueAxpr(name));
      out_anf_exprs.emplace_back(anf_expr);
    }
    op_pattern_ctx->Attr(op_unique_name)
        .Call(ctx->Var(axpr::kBuiltinList()).Apply(in_anf_exprs),
              ctx->Var(axpr::kBuiltinList()).Apply(out_anf_exprs));
    return adt::Ok{};
  };
  ADT_RETURN_IF_ERR(
      VisitEachSrcPtnCtxOpValueConnection(src_ptn_ctx, BuildConnection));
  return adt::Ok{};
}

}  // namespace ap::reified_drr
