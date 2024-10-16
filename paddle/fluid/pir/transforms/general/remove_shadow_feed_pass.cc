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

#include "paddle/fluid/pir/transforms/general/remove_shadow_feed_pass.h"

#include "paddle/common/enforce.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/dialect/operator/utils/op_yaml_info_parser.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/pir/include/core/block.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/core/ir_context.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"
#include "paddle/pir/include/pattern_rewrite/pattern_match.h"

namespace {

std::unique_ptr<paddle::dialect::OpYamlInfoParser> GetParser(
    pir::Operation *op) {
  std::unique_ptr<paddle::dialect::OpYamlInfoParser> op_info_parser(nullptr);
  std::string op_name = op->dyn_cast<paddle::dialect::PhiKernelOp>().op_name();
  auto op_info = pir::IrContext::Instance()->GetRegisteredOpInfo(op_name);
  if (op_info.HasInterface<paddle::dialect::OpYamlInfoInterface>()) {
    auto impl =
        op_info.GetInterfaceImpl<paddle::dialect::OpYamlInfoInterface>();
    auto op_info_tuple = impl->get_op_info_(op_name);
    op_info_parser = std::make_unique<paddle::dialect::OpYamlInfoParser>(
        op_info_tuple, paddle::dialect::IsLegacyOp(op_name));
  }
  return op_info_parser;
}

template <typename T>
phi::Place GetVarPlace(const paddle::framework::Variable *var,
                       const phi::Place &exe_place) {
  phi::Place place;
  auto &tensor = var->Get<T>();
  if (tensor.initialized()) {
    place = tensor.place();
  } else {
    place = exe_place;
  }
  return place;
}

class RemoveShadowFeedPattern
    : public pir::OpRewritePattern<paddle::dialect::PhiKernelOp> {
 public:
  explicit RemoveShadowFeedPattern(pir::IrContext *context,
                                   const pir::Block *block,
                                   const phi::Place &place,
                                   const paddle::framework::Scope *scope)
      : pir::OpRewritePattern<paddle::dialect::PhiKernelOp>::OpRewritePattern(
            context),
        place_(place),
        scope_(scope),
        kwargs_map_() {
    for (auto &[name, value] : block->kwargs()) {
      kwargs_map_[value] = name;
    }
  }

  bool IsSamePlaceShadowFeed(paddle::dialect::PhiKernelOp op) const {
    if (op.op_name() == "pd_op.shadow_feed") {
      auto in = op.operand_source(0);
      if (!kwargs_map_.count(in)) {
        return false;
      }
      auto in_name = kwargs_map_.at(in);
      auto *var = scope_->FindVar(in_name);
      if (!var) {
        return false;
      }
      phi::Place var_place, dst_place;
      if (var->IsType<phi::DenseTensor>()) {
        var_place = GetVarPlace<phi::DenseTensor>(var, place_);
      } else if (var->IsType<phi::SelectedRows>()) {
        var_place = GetVarPlace<phi::SelectedRows>(var, place_);
      } else if (var->IsType<paddle::framework::VariableRefArray>()) {
        var_place =
            GetVarPlace<paddle::framework::VariableRefArray>(var, place_);
      } else {
        PADDLE_THROW(common::errors::InvalidArgument(
            "RemoveShadowFeedPattern only support output "
            "variable of type DenseTensor, SelectedRows or VariableRefArray"));
      }

      int dst_place_type =
          op.attribute("dst_place_type").dyn_cast<pir::Int32Attribute>().data();
      if (dst_place_type == 0) {
        dst_place = phi::CPUPlace();
      } else {
        dst_place = place_;
      }

      return var_place == dst_place;
    }
    return false;
  }

  bool IsTensorAttrShadowFeed(paddle::dialect::PhiKernelOp op) const {
    if (op.op_name() == "pd_op.shadow_feed") {
      auto in = op.operand_source(0);
      if (!kwargs_map_.count(in)) {
        return false;
      }
      auto out = op.result(0);
      if (out.use_count() == 1) {
        auto use_op = out.first_use().owner();
        if (!use_op->isa<paddle::dialect::PhiKernelOp>()) {
          return false;
        }
        auto op_info_parser = GetParser(use_op);
        for (size_t i = 0; i < use_op->num_operands(); ++i) {
          if (out == use_op->operand_source(i) &&
              op_info_parser->IsTensorAttribute(i)) {
            return true;
          }
        }
      }
    }
    return false;
  }

  bool Match(paddle::dialect::PhiKernelOp op) const override {
    return IsSamePlaceShadowFeed(op) || IsTensorAttrShadowFeed(op);
  }

  void Rewrite(paddle::dialect::PhiKernelOp op,
               pir::PatternRewriter &rewriter) const override {  // NOLINT
    auto in = op.operand_source(0);
    auto out = op.result(0);
    in.set_type(out.type());
    rewriter.ReplaceAllUsesWith(out, in);
    rewriter.EraseOp(op);
  }

 private:
  const phi::Place place_;
  const paddle::framework::Scope *scope_;
  std::unordered_map<::pir::Value, std::string> kwargs_map_;
};

class RemoveShadowFeedPatternInference
    : public pir::OpRewritePattern<paddle::dialect::PhiKernelOp> {
 public:
  explicit RemoveShadowFeedPatternInference(pir::IrContext *context)
      : pir::OpRewritePattern<paddle::dialect::PhiKernelOp>::OpRewritePattern(
            context) {}

  bool Match(paddle::dialect::PhiKernelOp op) const override {
    return op.op_name() == "pd_op.shadow_feed";
  }

  void Rewrite(paddle::dialect::PhiKernelOp op,
               pir::PatternRewriter &rewriter) const override {  // NOLINT
    auto in = op.operand_source(0);
    auto out = op.result(0);
    in.set_type(out.type());
    rewriter.ReplaceAllUsesWith(out, in);
    rewriter.EraseOp(op);
  }
};

class RemoveShadowFeedPass : public pir::PatternRewritePass {
 public:
  RemoveShadowFeedPass()
      : pir::PatternRewritePass("remove_shadow_feed_pass", 0) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    if (Has("used_for_inference") && Get<bool>("used_for_inference")) {
      ps.Add<RemoveShadowFeedPatternInference>(context);
    } else {
      PADDLE_ENFORCE_EQ(
          Has("top_block"),
          true,
          common::errors::InvalidArgument(
              "Pass initialize failed."
              "When using RemoveShadowFeedPass, block attribute is required!"
              "Use Set method to set the place attribute."));
      PADDLE_ENFORCE_EQ(
          Has(pir::Pass::kPlaceAttr),
          true,
          common::errors::InvalidArgument(
              "Pass initialize failed."
              "When using RemoveShadowFeedPass, place attribute is required!"
              "Use Set method to set the place attribute."));
      PADDLE_ENFORCE_EQ(
          Has(pir::Pass::kParamScopeAttr),
          true,
          common::errors::InvalidArgument(
              "Pass initialize failed."
              "When using RemoveShadowFeedPass, scope attribute is required!"
              "Use Set method to set the scope attribute."));
      auto block = &Get<const pir::Block>("top_block");
      auto &place = Get<const phi::Place>(pir::Pass::kPlaceAttr);
      auto scope =
          &Get<const paddle::framework::Scope>(pir::Pass::kParamScopeAttr);
      PADDLE_ENFORCE_NOT_NULL(
          block, common::errors::InvalidArgument("block can not be nullptr"));
      PADDLE_ENFORCE_NOT_NULL(
          scope, common::errors::InvalidArgument("scope can not be nullptr"));

      ps.Add<RemoveShadowFeedPattern>(context, block, place, scope);
    }
    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<pir::Pass> CreateRemoveShadowFeedPass() {
  return std::make_unique<RemoveShadowFeedPass>();
}

}  // namespace pir
