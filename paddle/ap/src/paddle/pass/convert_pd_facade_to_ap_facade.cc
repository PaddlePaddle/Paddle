// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/ap/include/paddle/pass/convert_pd_facade_to_ap_facade.h"
#include "paddle/ap/include/paddle/hlir/manual_op.h"

#include "paddle/ap/include/axpr/abstract_list.h"
#include "paddle/ap/include/axpr/anf_expr_util.h"
#include "paddle/ap/include/axpr/atomic.h"
#include "paddle/ap/include/axpr/builtin_frame_util.h"
#include "paddle/ap/include/axpr/builtin_serializable_attr_map_to_axpr_helper.h"
#include "paddle/ap/include/axpr/data_type_util.h"
#include "paddle/ap/include/axpr/interpreter.h"
#include "paddle/ap/include/axpr/lambda_expr_builder.h"
#include "paddle/ap/include/paddle/builtin_frame_util.h"
#include "paddle/ap/include/paddle/pir/ap_pir_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace cinn::dialect::ir {

namespace adt = ap::adt;

namespace {

class ConvertPdFacadeToApFacadePattern
    : public pir::OpRewritePattern<paddle::dialect::ApFacadeOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::ApFacadeOp>::OpRewritePattern;

  bool MatchAndRewrite(paddle::dialect::ApFacadeOp pd_facade_op,
                       pir::PatternRewriter& rewriter) const override {
    const auto& ret = TryMatchAndRewrite(pd_facade_op, &rewriter);
    PADDLE_ENFORCE_EQ(
        ret.HasError(),
        false,
        phi::errors::Fatal(
            "ConvertPdFacadeToApFacadePattern::MatchAndRewrite failed. "
            "\nTraceback (most recent call "
            "last):\n%s\n%s: %s. ",
            ret.GetError().CallStackToString(),
            ret.GetError().class_name(),
            ret.GetError().msg()));
    return ret.GetOkValue();
  }

  adt::Result<bool> TryMatchAndRewrite(paddle::dialect::ApFacadeOp pd_facade_op,
                                       pir::PatternRewriter* rewriter) const {
    std::vector<pir::Value> inputs{};
    pir::Operation* upstream_op = nullptr;
    if (pd_facade_op->operand_source(0)) {
      upstream_op = pd_facade_op->operand_source(0).defining_op();
      ADT_CHECK(upstream_op != nullptr);
      ADT_CHECK(upstream_op->isa<pir::CombineOp>()) << adt::errors::TypeError{
          "the upstream of pd_op.ap_facade should builtin.combine"};
      inputs = upstream_op->dyn_cast<pir::CombineOp>().inputs();
    }
    ADT_CHECK(pd_facade_op->result(0).use_count() == 1);
    auto* downstream_op = pd_facade_op->result(0).first_use().owner();
    ADT_CHECK(downstream_op != nullptr);
    ADT_CHECK(downstream_op->isa<pir::SplitOp>()) << adt::errors::TypeError{
        "the downstream of pd_op.ap_facade should builtin.split"};
    ADT_LET_CONST_REF(attributes, GetFacadeOpAttributes(pd_facade_op));
    const auto old_outputs = downstream_op->dyn_cast<pir::SplitOp>().outputs();
    std::vector<pir::Type> output_types{};
    output_types.reserve(old_outputs.size());
    for (const auto& output : old_outputs) {
      output_types.emplace_back(output.type());
    }
    auto ap_facade_op = rewriter->Build<ap::dialect::FacadeOp>(
        inputs, attributes, output_types);
    for (int i = 0; i < old_outputs.size(); ++i) {
      rewriter->ReplaceAllUsesWith(old_outputs.at(i), ap_facade_op->result(i));
    }
    rewriter->EraseOp(downstream_op);
    rewriter->EraseOp(pd_facade_op);
    if (upstream_op != nullptr) {
      rewriter->EraseOp(upstream_op);
    }
    return true;
  }

  adt::Result<pir::AttributeMap> GetFacadeOpAttributes(
      paddle::dialect::ApFacadeOp pd_facade_op) const {
    ADT_LET_CONST_REF(serialized_attributes,
                      GetFacadeOpSerializedAttributes(pd_facade_op));
    ADT_LET_CONST_REF(lambda, CastStrToLambda(serialized_attributes));
    ADT_LET_CONST_REF(attr_map, RunLambda(lambda));
    return CastToPirAttributeMap(pd_facade_op, attr_map, serialized_attributes);
  }

  adt::Result<std::string> GetFacadeOpSerializedAttributes(
      paddle::dialect::ApFacadeOp op) const {
    const auto& iter = op->attributes().find("serialized_attributes");
    ADT_CHECK(iter != op->attributes().end());
    ADT_CHECK(iter->second.template isa<pir::StrAttribute>());
    return iter->second.template dyn_cast<pir::StrAttribute>().AsString();
  }

  adt::Result<ap::axpr::Lambda<ap::axpr::CoreExpr>> CastStrToLambda(
      const std::string& serialized_attributes) const {
    ADT_LET_CONST_REF(
        anf_expr, ap::axpr::MakeAnfExprFromJsonString(serialized_attributes));
    const auto& core_expr = ap::axpr::ConvertAnfExprToCoreExpr(anf_expr);
    std::vector<ap::axpr::tVar<std::string>> args{};
    return ap::axpr::Lambda<ap::axpr::CoreExpr>{args, core_expr};
  }

  adt::Result<ap::axpr::AttrMap<ap::axpr::Value>> RunLambda(
      const ap::axpr::Lambda<ap::axpr::CoreExpr>& lambda) const {
    ap::memory::Guard guard{};
    ap::axpr::Interpreter interpreter(
        ap::paddle::MakeBuiltinFrameAttrMap<ap::axpr::Value>(),
        guard.circlable_ref_list());
    ADT_LET_CONST_REF(ret_val, interpreter.Interpret(lambda, {}));
    ADT_LET_CONST_REF(
        attr_map_val,
        ret_val.template CastTo<ap::axpr::AttrMap<ap::axpr::Value>>());
    return attr_map_val;
  }

  adt::Result<pir::AttributeMap> CastToPirAttributeMap(
      paddle::dialect::ApFacadeOp pd_facade_op,
      const ap::axpr::AttrMap<ap::axpr::Value>& attr_map,
      const std::string& serialized_attributes) const {
    pir::AttributeMap attributes{};
    for (const auto& [name, val] : attr_map->storage) {
      ADT_LET_CONST_REF(pir_attr, CastToPirAttribute(val));
      attributes[name] = pir_attr;
    }
    const auto& CopyAttribute =
        [&](const auto& attr_name) -> adt::Result<adt::Ok> {
      const auto& iter = pd_facade_op->attributes().find(attr_name);
      ADT_CHECK(iter != pd_facade_op->attributes().end());
      attributes[attr_name] = iter->second;
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(CopyAttribute("custom_op_name"));
    ADT_RETURN_IF_ERR(CopyAttribute("infer_meta_func_name"));
    ADT_RETURN_IF_ERR(CopyAttribute("infer_symbolic_func_name"));
    attributes["__original_serialized_attributes__"] = pir::StrAttribute::get(
        pir::IrContext::Instance(), serialized_attributes);
    return attributes;
  }

  adt::Result<pir::Attribute> CastToPirAttribute(
      const ap::axpr::Value& val) const {
    ADT_LET_CONST_REF(ap_pir_attr, ap::dialect::ApPirAttribute::CastFrom(val));
    return ap_pir_attr.CastToPirAttribute();
  }
};

class ConvertPdFacadeToApFacadePass : public pir::PatternRewritePass {
 public:
  ConvertPdFacadeToApFacadePass()
      : pir::PatternRewritePass("convert_pd_facade_to_ap_facade_pass", 1) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext* context) override {
    pir::RewritePatternSet ps(context);
    ps.Add<ConvertPdFacadeToApFacadePattern>(context);
    return ps;
  }
};

}  // namespace

std::unique_ptr<::pir::Pass> CreateConvertPdFacadeToApFacadePass() {
  return std::make_unique<ConvertPdFacadeToApFacadePass>();
}

}  // namespace cinn::dialect::ir
