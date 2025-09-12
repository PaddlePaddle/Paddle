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

#include "paddle/ap/include/reified_drr/reified_res_ptn_axpr_maker.h"
#include "paddle/ap/include/code_module/module_to_axpr_helper.h"
#include "paddle/ap/include/drr/node.h"
#include "paddle/ap/include/drr/op_tensor_pattern_ctx_helper.h"

namespace ap::reified_drr {

namespace {

using ResPtnCtxOpImpl =
    std::variant<drr::NativeIrOp<drr::Node>, drr::PackedIrOp<drr::Node>>;

struct ResPtnCtxOp : public ResPtnCtxOpImpl {
  using ResPtnCtxOpImpl::ResPtnCtxOpImpl;
  ADT_DEFINE_VARIANT_METHODS(ResPtnCtxOpImpl);

  const std::string& op_unique_name() const {
    using RetT = const std::string&;
    return Match([&](const auto& impl) -> RetT { return impl->name; });
  }

  static std::optional<ResPtnCtxOp> CastFromDrrNode(const drr::Node& drr_node) {
    using RetT = std::optional<ResPtnCtxOp>;
    return drr_node.Match(
        [&](const drr::NativeIrOp<drr::Node>& impl) -> RetT { return impl; },
        [&](const drr::PackedIrOp<drr::Node>& impl) -> RetT { return impl; },
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
    const std::function<adt::Result<code_gen::CodeGenResult<axpr::Value>>(
        const std::string&)>& CodeGenResult4FusedOpName,
    const std::string& op_unique_name,
    const drr::NativeIrOp<drr::Node>& ir_op) {
  const auto& op =
      op_pattern_ctx->Attr("ap_native_op").Call(ir_op->op_declare->op_name);
  op_pattern_ctx->SetAttr(op_unique_name, op);
  return adt::Ok{};
}

adt::Result<axpr::AnfExpr> GenCodeGenLambda(
    const std::function<adt::Result<code_gen::CodeGenResult<axpr::Value>>(
        const std::string&)>& CodeGenResult4FusedOpName,
    const drr::PackedIrOp<drr::Node>& ir_op) {
  ADT_LET_CONST_REF(code_gen_result, CodeGenResult4FusedOpName(ir_op->name));
  axpr::LambdaExprBuilder lmd{};
  auto GetBody = [&](auto& ctx) -> adt::Result<axpr::AnfExpr> {
    ADT_LET_CONST_REF(code_module_anf_expr,
                      code_module::ModuleToAxprHelper{}.ConvertModuleToAnfExpr(
                          &ctx, code_gen_result->code_module));
    const auto& kernel_dispatch_lambda_anf_expr =
        axpr::ConvertCoreExprToAnfExpr(
            code_gen_result->kernel_dispatch_func->lambda);
    const auto& kernel_dispatch_func_name = ctx.NewTmpVarName();
    ctx.Var(kernel_dispatch_func_name) = kernel_dispatch_lambda_anf_expr;
    const auto& kernel_dispatch_func_anf_expr =
        ctx.Var(kernel_dispatch_func_name).Attr("__function__");
    ADT_LET_CONST_REF(kernel_dispatch_const_data_anf_expr,
                      axpr::BuiltinSerializableAttrMapToAxprHelper{}.Convert(
                          &ctx, code_gen_result->kernel_dispatch_const_data));
    std::map<std::string, axpr::AnfExpr> kwargs{
        {"module", code_module_anf_expr},
        {"kernel_dispatch_func", kernel_dispatch_func_anf_expr},
        {"kernel_dispatch_const_data", kernel_dispatch_const_data_anf_expr},
    };
    return ctx.Var("CodeGenResult").Apply(std::vector<axpr::AnfExpr>{}, kwargs);
  };
  ADT_LET_CONST_REF(ret, lmd.TryLambda({"ctx", "o", "t"}, GetBody));
  return ret;
}

adt::Result<adt::Ok> GenAnfExprForOpImpl(
    axpr::LetVar* op_pattern_ctx,
    const std::function<adt::Result<code_gen::CodeGenResult<axpr::Value>>(
        const std::string&)>& CodeGenResult4FusedOpName,
    const std::string& op_unique_name,
    const drr::PackedIrOp<drr::Node>& ir_op) {
  ADT_LET_CONST_REF(code_gen_lambda,
                    GenCodeGenLambda(CodeGenResult4FusedOpName, ir_op));
  auto* ctx = op_pattern_ctx->ctx();
  const std::string& lambda_name = ctx->NewTmpVarName();
  ctx->Var(lambda_name) = code_gen_lambda;
  const auto& code_gen_func = ctx->Var(lambda_name);
  const auto& op =
      op_pattern_ctx->Attr("ap_pattern_fusion_op").Call(code_gen_func);
  op_pattern_ctx->SetAttr(op_unique_name, op);
  return adt::Ok{};
}

template <typename DoEachT>
adt::Result<adt::Ok> VisitEachResPtnCtxOp(
    const drr::ResultPatternCtx& res_ptn_ctx, const DoEachT& DoEach) {
  for (const auto& node : res_ptn_ctx->node_arena->nodes()) {
    const auto& res_ptn_ctx_op = ResPtnCtxOp::CastFromDrrNode(node);
    if (res_ptn_ctx_op.has_value()) {
      ADT_RETURN_IF_ERR(DoEach(res_ptn_ctx_op.value()));
    }
  }
  return adt::Ok{};
}

}  // namespace

adt::Result<adt::Ok> ReifiedResPtnAxprMaker::GenAnfExprForResPtnCtxOps(
    axpr::LetVar* op_pattern_ctx) {
  using Ok = adt::Result<adt::Ok>;
  auto GenAnfExprForOp = [&](const ResPtnCtxOp& op) -> Ok {
    const auto& op_unique_name = op.op_unique_name();
    return op.Match([&](const auto& impl) -> Ok {
      return GenAnfExprForOpImpl(
          op_pattern_ctx, CodeGenResult4FusedOpName_, op_unique_name, impl);
    });
  };
  ADT_RETURN_IF_ERR(VisitEachResPtnCtxOp(res_ptn_ctx_, GenAnfExprForOp));
  return adt::Ok{};
}

namespace {

template <typename DoEachT>
adt::Result<adt::Ok> VisitEachResPtnCtxOpValueConnection(
    const drr::ResultPatternCtx& res_ptn_ctx, const DoEachT& DoEach) {
  auto DoEachOp = [&](const ResPtnCtxOp& op) -> adt::Result<adt::Ok> {
    ADT_LET_CONST_REF(input_value_names, op.GetInputValueNames());
    ADT_LET_CONST_REF(output_value_names, op.GetOutputValueNames());
    ADT_RETURN_IF_ERR(
        DoEach(op.op_unique_name(), input_value_names, output_value_names));
    return adt::Ok{};
  };
  ADT_RETURN_IF_ERR(VisitEachResPtnCtxOp(res_ptn_ctx, DoEachOp));
  return adt::Ok{};
}

}  // namespace

adt::Result<adt::Ok>
ReifiedResPtnAxprMaker::GenAnfExprForResPtnCtxOpValueConnections(
    axpr::LetVar* op_pattern_ctx, axpr::LetVar* tensor_pattern_ctx) {
  ADT_CHECK(op_pattern_ctx->ctx() == tensor_pattern_ctx->ctx());
  auto* ctx = op_pattern_ctx->ctx();
  using Ok = adt::Result<adt::Ok>;
  auto GetDrrIrValueAxpr =
      [&](const auto& tensor_name) -> adt::Result<axpr::AnfExpr> {
    ADT_LET_CONST_REF(ir_value,
                      drr::OpTensorPatternCtxHelper{}.GetIrValueByUid(
                          res_ptn_ctx_->tensor_pattern_ctx, tensor_name));
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
      VisitEachResPtnCtxOpValueConnection(res_ptn_ctx_, BuildConnection));
  return adt::Ok{};
}

}  // namespace ap::reified_drr
