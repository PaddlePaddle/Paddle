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

#include "paddle/ap/include/paddle/pir/pir_node_matched_src_ptn_ctx_helper.h"
#include <memory>
#include "glog/logging.h"
#include "paddle/ap/include/axpr/anf_expr_util.h"
#include "paddle/ap/include/axpr/cps_interpreter.h"
#include "paddle/ap/include/axpr/lambda_expr_builder.h"
#include "paddle/ap/include/drr/builtin_frame_util.h"
#include "paddle/ap/include/drr/drr_node_descriptor.h"
#include "paddle/ap/include/drr/ir_op.h"
#include "paddle/ap/include/drr/ir_value.h"
#include "paddle/ap/include/drr/node.h"
#include "paddle/ap/include/drr/source_pattern_ctx.h"
#include "paddle/ap/include/drr/src_ptn_op_pattern_ctx_method_class.h"
#include "paddle/ap/include/drr/src_ptn_tensor_pattern_ctx_method_class.h"
#include "paddle/ap/include/drr/tags.h"
#include "paddle/ap/include/drr/value_method_class.h"
#include "paddle/ap/include/graph/node.h"
#include "paddle/ap/include/paddle/pir/attribute_method_class.h"
#include "paddle/ap/include/paddle/pir/packed_ir_op_inner_source_pattern_helper.h"
#include "paddle/ap/include/paddle/pir/pir_method_class.h"
#include "paddle/ap/include/paddle/pir/type_method_class.h"
#include "paddle/pir/include/core/block.h"
#include "paddle/pir/include/core/operation.h"

namespace ap::paddle {

namespace {

adt::Result<std::function<adt::Result<std::string>(const pir::Operation*)>>
MakeOpNameGetter(const pir::Block* block) {
  using CacheT = std::map<const pir::Operation*, std::string>;
  CacheT cache{};
  {
    int i = 0;
    for (auto& op : *block) {
      ADT_CHECK(cache.emplace(&op, op.name() + "_" + std::to_string(i)).second);
      ++i;
    }
  }
  using RetFunc =
      std::function<adt::Result<std::string>(const pir::Operation*)>;

  RetFunc func = [cache = std::move(cache)](
                     const pir::Operation* op) -> adt::Result<std::string> {
    const auto& iter = cache.find(op);
    ADT_CHECK(iter != cache.end());
    return iter->second;
  };
  return func;
}

adt::Result<std::function<adt::Result<std::string>(pir::Value)>>
MakeTensorNameGetter(const pir::Block* block) {
  using CacheT = std::map<pir::Value, std::string>;
  CacheT cache{};
  {
    int i = 0;
    for (auto& op : *block) {
      for (int j = 0; j < op.num_results(); ++j) {
        pir::Value value = op.result(j);
        const auto& name =
            op.name() + "_" + std::to_string(i) + "_" + std::to_string(j);
        ADT_CHECK(cache.emplace(value, name).second);
      }
      ++i;
    }
  }
  {
    int i = 0;
    for (auto& op : *block) {
      for (int j = 0; j < op.num_operands(); ++j) {
        pir::Value value = op.operand_source(j);
        if (cache.count(value) > 0) {
          continue;
        }
        const auto& name = std::string() + "input_" + std::to_string(i++);
        ADT_CHECK(cache.emplace(value, name).second);
      }
    }
  }
  using RetFunc = std::function<adt::Result<std::string>(pir::Value)>;
  RetFunc func =
      [cache = std::move(cache)](pir::Value value) -> adt::Result<std::string> {
    const auto& iter = cache.find(value);
    ADT_CHECK(iter != cache.end());
    return iter->second;
  };
  return func;
}

class SourcePatternCtxBuilder {
  drr::SourcePatternCtx src_ptn_ctx_;
  std::unique_ptr<axpr::CpsInterpreter> interpreter_;

 public:
  SourcePatternCtxBuilder(const drr::SourcePatternCtx& src_ptn_ctx,
                          std::unique_ptr<axpr::CpsInterpreter>&& interpreter)
      : src_ptn_ctx_(src_ptn_ctx), interpreter_(std::move(interpreter)) {}

  const drr::SourcePatternCtx& src_ptn_ctx() const { return src_ptn_ctx_; }

  adt::Result<adt::Ok> BuildNativeOp(const std::string& op_name,
                                     const std::string& op_unique_name) {
    static const axpr::Lambda<axpr::CoreExpr> func([]() {
      axpr::LambdaExprBuilder lmd;
      const auto& anf_expr = lmd.Lambda(
          {"o", "op_name", "op_unique_name"},
          [&](axpr::LetContext& ctx) -> axpr::AnfExpr {
            auto& native_op =
                ctx.Var("o").Attr("ap_native_op").Call(ctx.Var("op_name"));
            ctx.Var("o").SetAttr(ctx.Var("op_unique_name"), native_op);
            return ctx.None();
          });
      const auto& core_expr = axpr::ConvertAnfExprToCoreExpr(anf_expr);
      CHECK(core_expr.Has<axpr::Atomic<axpr::CoreExpr>>());
      const auto& atomic = core_expr.Get<axpr::Atomic<axpr::CoreExpr>>();
      CHECK(atomic.Has<axpr::Lambda<axpr::CoreExpr>>());
      return atomic.Get<axpr::Lambda<axpr::CoreExpr>>();
    }());
    const auto& op_pattern_ctx = drr::GetSrcPtnOpPatternCtxClass().New(
        drr::SrcPtn(src_ptn_ctx_->op_pattern_ctx));
    ADT_RETURN_IF_ERR(interpreter_->Interpret(func,
                                              {axpr::Value{op_pattern_ctx},
                                               axpr::Value{op_name},
                                               axpr::Value{op_unique_name}}));
    return adt::Ok{};
  }

  adt::Result<adt::Ok> SetOpAttr(const std::string& op_unique_name,
                                 const std::string& attr_name,
                                 pir::Attribute attribute) {
    static const axpr::Lambda<axpr::CoreExpr> func([]() {
      axpr::LambdaExprBuilder lmd;
      const auto& anf_expr =
          lmd.Lambda({"o", "op_unique_name", "attr_name", "attr_val"},
                     [&](axpr::LetContext& ctx) -> axpr::AnfExpr {
                       ctx.Var("o")
                           .Attr(ctx.Var("op_unique_name"))
                           .SetAttr(ctx.Var("attr_name"), ctx.Var("attr_val"));
                       return ctx.None();
                     });
      const auto& core_expr = axpr::ConvertAnfExprToCoreExpr(anf_expr);
      CHECK(core_expr.Has<axpr::Atomic<axpr::CoreExpr>>());
      const auto& atomic = core_expr.Get<axpr::Atomic<axpr::CoreExpr>>();
      CHECK(atomic.Has<axpr::Lambda<axpr::CoreExpr>>());
      return atomic.Get<axpr::Lambda<axpr::CoreExpr>>();
    }());
    const auto& op_pattern_ctx = drr::GetSrcPtnOpPatternCtxClass().New(
        drr::SrcPtn(src_ptn_ctx_->op_pattern_ctx));
    const auto& attr_val = GetPirAttributeClass().New(attribute);
    ADT_RETURN_IF_ERR(interpreter_->Interpret(func,
                                              {axpr::Value{op_pattern_ctx},
                                               axpr::Value{op_unique_name},
                                               axpr::Value{attr_name},
                                               axpr::Value{attr_val}}));
    return adt::Ok{};
  }

  adt::Result<adt::Ok> SetTensorType(const std::string& tensor_unique_name,
                                     pir::Type type) {
    static const axpr::Lambda<axpr::CoreExpr> func([]() {
      axpr::LambdaExprBuilder lmd;
      const auto& anf_expr =
          lmd.Lambda({"t", "tensor_unique_name", "type_val"},
                     [&](axpr::LetContext& ctx) -> axpr::AnfExpr {
                       ctx.Var("t")
                           .Attr(ctx.Var("tensor_unique_name"))
                           .SetAttr("type", ctx.Var("type_val"));
                       return ctx.None();
                     });
      const auto& core_expr = axpr::ConvertAnfExprToCoreExpr(anf_expr);
      CHECK(core_expr.Has<axpr::Atomic<axpr::CoreExpr>>());
      const auto& atomic = core_expr.Get<axpr::Atomic<axpr::CoreExpr>>();
      CHECK(atomic.Has<axpr::Lambda<axpr::CoreExpr>>());
      return atomic.Get<axpr::Lambda<axpr::CoreExpr>>();
    }());
    const auto& tensor_pattern_ctx = drr::GetSrcPtnTensorPatternCtxClass().New(
        drr::SrcPtn(src_ptn_ctx_->tensor_pattern_ctx));
    auto GetType = [&]() -> axpr::Value { return GetPirTypeClass().New(type); };
    ADT_RETURN_IF_ERR(interpreter_->Interpret(func,
                                              {axpr::Value{tensor_pattern_ctx},
                                               axpr::Value{tensor_unique_name},
                                               axpr::Value{GetType()}}));
    return adt::Ok{};
  }

  adt::Result<adt::Ok> Connect(
      const std::string& op_unique_name,
      const std::vector<std::optional<std::string>>& input_tensor_names,
      const std::vector<std::string>& output_tensor_names) {
    static const axpr::Lambda<axpr::CoreExpr> func([]() {
      axpr::LambdaExprBuilder lmd;
      const auto& anf_expr = lmd.Lambda(
          {"o",
           "t",
           "op_unique_name",
           "input_tensor_names_val",
           "output_tensor_names_val"},
          [&](axpr::LetContext& ctx) -> axpr::AnfExpr {
            const auto& get_or_create =
                ctx.Var("t").Attr("get_or_create_tensor");
            auto& op = ctx.Var("o").Attr(ctx.Var("op_unique_name"));
            op.Call(ctx.Var("map").Call(get_or_create,
                                        ctx.Var("input_tensor_names_val")),
                    ctx.Var("map").Call(get_or_create,
                                        ctx.Var("output_tensor_names_val")));
            return ctx.None();
          });
      const auto& core_expr = axpr::ConvertAnfExprToCoreExpr(anf_expr);
      CHECK(core_expr.Has<axpr::Atomic<axpr::CoreExpr>>());
      const auto& atomic = core_expr.Get<axpr::Atomic<axpr::CoreExpr>>();
      CHECK(atomic.Has<axpr::Lambda<axpr::CoreExpr>>());
      return atomic.Get<axpr::Lambda<axpr::CoreExpr>>();
    }());
    const auto& op_pattern_ctx = drr::GetSrcPtnOpPatternCtxClass().New(
        drr::SrcPtn(src_ptn_ctx_->op_pattern_ctx));
    const auto& tensor_pattern_ctx = drr::GetSrcPtnTensorPatternCtxClass().New(
        drr::SrcPtn(src_ptn_ctx_->tensor_pattern_ctx));
    const auto& input_tensor_names_val = OptStrsToList(input_tensor_names);
    const auto& output_tensor_names_val = StrsToList(output_tensor_names);
    ADT_RETURN_IF_ERR(
        interpreter_->Interpret(func,
                                {
                                    axpr::Value{op_pattern_ctx},
                                    axpr::Value{tensor_pattern_ctx},
                                    axpr::Value{op_unique_name},
                                    axpr::Value{input_tensor_names_val},
                                    axpr::Value{output_tensor_names_val},
                                }));
    return adt::Ok{};
  }

 private:
  adt::List<axpr::Value> StrsToList(const std::vector<std::string>& strs) {
    adt::List<axpr::Value> ret;
    ret->reserve(strs.size());
    for (const auto& str : strs) {
      ret->emplace_back(axpr::Value{str});
    }
    return ret;
  }

  adt::List<axpr::Value> OptStrsToList(
      const std::vector<std::optional<std::string>>& strs) {
    adt::List<axpr::Value> ret;
    ret->reserve(strs.size());
    for (const auto& str : strs) {
      if (str.has_value()) {
        ret->emplace_back(axpr::Value{str.value()});
      } else {
        ret->emplace_back(adt::Nothing{});
      }
    }
    return ret;
  }
};

std::unique_ptr<SourcePatternCtxBuilder> MakeSourcePatternCtxBuilder(
    const std::shared_ptr<drr::DrrCtxImpl>& drr_ctx) {
  auto node_arena = std::make_shared<graph::NodeArena<drr::Node>>();
  drr::SourcePatternCtx src_ptn_ctx{
      node_arena,
      drr::OpPatternCtx{
          node_arena, std::map<std::string, drr::IrOp>{}, drr_ctx},
      drr::TensorPatternCtx{
          node_arena, std::map<std::string, drr::IrValue>{}, drr_ctx}};
  const auto& builtin_frame = ap::drr::MakeBuiltinFrameAttrMap();
  auto interpreter = std::make_unique<axpr::CpsInterpreter>(
      builtin_frame, drr_ctx->circlable_ref_list);
  return std::make_unique<SourcePatternCtxBuilder>(src_ptn_ctx,
                                                   std::move(interpreter));
}

adt::Result<adt::Ok> InitSrcPtnCtxNativeIrOps(
    SourcePatternCtxBuilder* builder,
    pir::Block* block,
    const std::function<adt::Result<std::string>(const pir::Operation*)>&
        GetOpName) {
  for (auto& op : *block) {
    auto* op_ptr = &op;
    ADT_LET_CONST_REF(op_unique_name, GetOpName(op_ptr));
    ADT_RETURN_IF_ERR(builder->BuildNativeOp(op.name(), op_unique_name));
    for (const auto& [attr_name, attr_val] : op.attributes()) {
      ADT_RETURN_IF_ERR(
          builder->SetOpAttr(op_unique_name, attr_name, attr_val));
    }
  }
  return adt::Ok{};
}

adt::Result<adt::Ok> InitSrcPtnCtxNativeIrValues(
    SourcePatternCtxBuilder* builder,
    pir::Block* block,
    const std::function<adt::Result<std::string>(pir::Value)>& GetTensorName) {
  std::unordered_set<pir::Value> inited;
  auto InitType = [&](pir::Value value) -> adt::Result<adt::Ok> {
    if (inited.insert(value).second) {
      ADT_LET_CONST_REF(tensor_name, GetTensorName(value));
      ADT_RETURN_IF_ERR(builder->SetTensorType(tensor_name, value.type()));
    }
    return adt::Ok{};
  };
  for (auto& op : *block) {
    for (int i = 0; i < op.num_operands(); ++i) {
      if (op.operand_source(i)) {
        ADT_RETURN_IF_ERR(InitType(op.operand_source(i)));
      }
    }
    for (int i = 0; i < op.num_results(); ++i) {
      ADT_RETURN_IF_ERR(InitType(op.result(i)));
    }
  }
  return adt::Ok{};
}

adt::Result<adt::Ok> InitSrcPtnCtxConnections(
    SourcePatternCtxBuilder* builder,
    pir::Block* block,
    const std::function<adt::Result<std::string>(const pir::Operation*)>&
        GetOpName,
    const std::function<adt::Result<std::string>(pir::Value)>& GetTensorName) {
  for (auto& op : *block) {
    auto* op_ptr = &op;
    ADT_LET_CONST_REF(op_unique_name, GetOpName(op_ptr));
    std::vector<std::optional<std::string>> input_tensor_names{};
    input_tensor_names.reserve(op.num_operands());
    for (int i = 0; i < op.num_operands(); ++i) {
      if (op.operand_source(i)) {
        ADT_LET_CONST_REF(input_name, GetTensorName(op.operand_source(i)));
        input_tensor_names.emplace_back(input_name);
      } else {
        input_tensor_names.emplace_back(std::nullopt);
      }
    }
    std::vector<std::string> output_tensor_names{};
    output_tensor_names.reserve(op.num_results());
    for (int i = 0; i < op.num_results(); ++i) {
      ADT_LET_CONST_REF(output_name, GetTensorName(op.result(i)));
      output_tensor_names.emplace_back(output_name);
    }
    ADT_RETURN_IF_ERR(builder->Connect(
        op_unique_name, input_tensor_names, output_tensor_names));
  }
  return adt::Ok{};
}

}  // namespace

adt::Result<std::shared_ptr<reified_drr::MatchedSrcPtnCtxHelper>>
PirNodeMatchedSrcPtnCtxHelper::MakeInnerMatchedSrcPtnCtxHelper(
    const drr::PackedIrOp<drr::Node>& drr_packed_ir_op) {
  ADT_LET_CONST_REF(pir_node,
                    match_ctx_->GetSoleBigGraphNode(drr_packed_ir_op->node));
  ADT_LET_CONST_REF(pir_packed_ir_op, pir_node.template TryGet<PackedIrOp>());
  ADT_LET_CONST_REF(
      op_pattern_ctx,
      adt::WeakPtrLock(drr_packed_ir_op->op_declare->op_pattern_ctx));
  ADT_LET_CONST_REF(drr_ctx, adt::WeakPtrLock(op_pattern_ctx->drr_ctx));
  ADT_LET_CONST_REF(
      src_ptn_ctx,
      ConvertBlockToSrcPtnCtx(pir_packed_ir_op.fusion_op.block(), drr_ctx));
  PackedIrOpInnerSourcePatternHelper inner_src_ptn_ctx_helper{};
  ADT_LET_CONST_REF(
      opt_match_ctx,
      inner_src_ptn_ctx_helper.Match(pir_packed_ir_op, src_ptn_ctx));
  ADT_CHECK(opt_match_ctx.has_value());
  std::shared_ptr<reified_drr::MatchedSrcPtnCtxHelper>
      matched_src_ptn_ctx_helper =
          std::make_shared<PirNodeMatchedSrcPtnCtxHelper>(
              src_ptn_ctx, opt_match_ctx.value());
  return matched_src_ptn_ctx_helper;
}

adt::Result<adt::Ok> PirNodeMatchedSrcPtnCtxHelper::VisitNativeIrOpAttr(
    const drr::NativeIrOp<drr::Node>& drr_native_ir_op,
    const std::function<adt::Result<adt::Ok>(const std::string& attr_name,
                                             const axpr::Value& attr_val)>&
        DoEachAttr) {
  ADT_LET_CONST_REF(pir_node,
                    match_ctx_->GetSoleBigGraphNode(drr_native_ir_op->node));
  ADT_LET_CONST_REF(pir_native_ir_op, pir_node.template TryGet<NativeIrOp>());
  for (const auto& [attr_name, attr] : pir_native_ir_op.op->attributes()) {
    if (!attr) continue;
    if (attr_name == "op_callstack") continue;
    if (attr_name == "sym_shape_str") continue;
    const auto& attr_val = GetPirAttributeClass().New(attr);
    ADT_RETURN_IF_ERR(DoEachAttr(attr_name, attr_val));
  }
  return adt::Ok{};
}

adt::Result<axpr::Value> PirNodeMatchedSrcPtnCtxHelper::GetNativeIrValueType(
    const drr::NativeIrValue<drr::Node>& native_ir_value) {
  ADT_LET_CONST_REF(pir_node,
                    match_ctx_->GetSoleBigGraphNode(native_ir_value->node));
  ADT_LET_CONST_REF(pir_native_ir_value,
                    pir_node.template TryGet<NativeIrValue>());
  return GetPirTypeClass().New(pir_native_ir_value.value.type());
}

adt::Result<drr::SourcePatternCtx>
PirNodeMatchedSrcPtnCtxHelper::ConvertBlockToSrcPtnCtx(
    pir::Block* block, const std::shared_ptr<drr::DrrCtxImpl>& drr_ctx) {
  ADT_LET_CONST_REF(GetOpName, MakeOpNameGetter(block));
  ADT_LET_CONST_REF(GetTensorName, MakeTensorNameGetter(block));
  auto builder = MakeSourcePatternCtxBuilder(drr_ctx);
  ADT_RETURN_IF_ERR(InitSrcPtnCtxNativeIrOps(builder.get(), block, GetOpName));
  ADT_RETURN_IF_ERR(
      InitSrcPtnCtxNativeIrValues(builder.get(), block, GetTensorName));
  ADT_RETURN_IF_ERR(
      InitSrcPtnCtxConnections(builder.get(), block, GetOpName, GetTensorName));
  return builder->src_ptn_ctx();
}

}  // namespace ap::paddle
