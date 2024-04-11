// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/core/ir_context.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"
#include "paddle/pir/include/pattern_rewrite/pattern_match.h"

namespace {

std::unique_ptr<paddle::dialect::OpYamlInfoParser> GetOpYamlInfoParser(
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

class RemoveShadowFeedPattern : public pir::RewritePattern {
 public:
  RemoveShadowFeedPattern(pir::IrContext *context,
                          const pir::Block *block,
                          const phi::Place &place,
                          const paddle::framework::Scope *scope)
      : RewritePattern(MatchAnyOpTypeTag(),
                       1 /*benefit*/,
                       context,
                       {} /*generated_names*/),
        block_(block),
        place_(place),
        scope_(scope) {
    for (auto &[name, value] : block->kwargs()) {
      kwargs_map_[value] = name;
    }
  }

  // bool MatchAndRewrite(pir::Operation *op,
  //                      pir::PatternRewriter &rewriter) const override {
  //   VLOG(1) << "RemoveShadowFeedPattern isa PhiKernelOp: "
  //           << op->isa<paddle::dialect::PhiKernelOp>();
  //   if (!op->isa<paddle::dialect::PhiKernelOp>()) {
  //     VLOG(1) << "RemoveShadowFeedPattern op name: " << op->name();
  //   }
  //   if (op->isa<paddle::dialect::PhiKernelOp>() &&
  //       op->dyn_cast<paddle::dialect::PhiKernelOp>().op_name() ==
  //           "pd_op.shadow_feed") {
  //     VLOG(1) << "RemoveShadowFeedPattern find pd_op.shadow_feed";
  //     auto in = op->operand_source(0);
  //     if (!kwargs_map_.count(in)) {
  //       VLOG(1) << "RemoveShadowFeedPattern kwargs_map_ not contain var!";
  //       return false;
  //     }
  //     auto in_name = kwargs_map_.at(in);
  //     VLOG(1) << "RemoveShadowFeedPattern in_name: " << in_name;
  //     auto *var = scope_->FindVar(in_name);
  //     phi::Place var_place;
  //     if (var->IsType<phi::DenseTensor>()) {
  //       auto &tensor = var->Get<phi::DenseTensor>();
  //       VLOG(1) << "RemoveShadowFeedPattern tensor.initialized(): "
  //               << tensor.initialized();
  //       if (tensor.initialized()) {
  //         var_place = tensor.place();
  //       } else {
  //         var_place = place_;
  //       }
  //     } else {
  //       PADDLE_THROW(paddle::platform::errors::InvalidArgument(
  //           "RemoveShadowFeedPattern only support output "
  //           "variable of type DenseTensor, SelectedRows or
  //           VariableRefArray"));
  //     }
  //     if (var_place == place_) {
  //       VLOG(1) << "RemoveShadowFeedPattern remove pd_op.shadow_feed";
  //       auto out = op->result(0);
  //       in.set_type(out.type());
  //       rewriter.ReplaceAllUsesWith(out, in);
  //       rewriter.EraseOp(op);
  //       return true;
  //     }
  //     return false;
  //   } else if (op->isa<paddle::dialect::PhiKernelOp>() &&
  //              op->dyn_cast<paddle::dialect::PhiKernelOp>().op_name() !=
  //                  "pd_op.shadow_feed") {
  //     VLOG(1) << "auto op_info_parser = GetOpYamlInfoParser(op)";
  //     auto op_info_parser = GetOpYamlInfoParser(op);
  //     VLOG(1) << "finish auto op_info_parser = GetOpYamlInfoParser(op), "
  //                "op->num_operands(): "
  //             << op->num_operands();
  //     for (size_t i = 0; i < op->num_operands(); ++i) {
  //       auto define_op = op->operand_source(i).defining_op();
  //       VLOG(1) << "define_op: " << define_op;
  //       VLOG(1) << "op_info_parser->IsTensorAttribute(i): "
  //               << op_info_parser->IsTensorAttribute(i);

  //       if (op_info_parser->IsTensorAttribute(i) && define_op &&
  //           define_op->isa<paddle::dialect::PhiKernelOp>() &&
  //           define_op->dyn_cast<paddle::dialect::PhiKernelOp>().op_name() ==
  //               "pd_op.shadow_feed") {
  //         auto in = define_op->operand_source(0);
  //         auto out = define_op->result(0);
  //         rewriter.ReplaceAllUsesWith(out, in);
  //         rewriter.EraseOp(define_op);
  //         VLOG(1) << "RemoveShadowFeedPattern remove IsTensorAttribute "
  //                    "shadow_feed: "
  //                 << op->dyn_cast<paddle::dialect::PhiKernelOp>().op_name();
  //         return true;
  //       }
  //     }
  //     return false;
  //   }
  //   return false;
  // }

  bool IsSamePlaceShadowFeed(pir::Operation *op) const {
    if (op->isa<paddle::dialect::PhiKernelOp>() &&
        op->dyn_cast<paddle::dialect::PhiKernelOp>().op_name() ==
            "pd_op.shadow_feed") {
      auto in = op->operand_source(0);
      if (!kwargs_map_.count(in)) {
        return false;
      }
      auto in_name = kwargs_map_.at(in);
      auto *var = scope_->FindVar(in_name);
      phi::Place var_place;
      if (var->IsType<phi::DenseTensor>()) {
        auto &tensor = var->Get<phi::DenseTensor>();
        if (tensor.initialized()) {
          var_place = tensor.place();
        } else {
          var_place = place_;
        }
      } else {
        PADDLE_THROW(paddle::platform::errors::InvalidArgument(
            "RemoveShadowFeedPattern only support output "
            "variable of type DenseTensor, SelectedRows or VariableRefArray"));
      }
      return var_place == place_;
    }
    return false;
  }

  bool IsTensorAttrShadowFeed(pir::Operation *op) const {
    if (op->isa<paddle::dialect::PhiKernelOp>() &&
        op->dyn_cast<paddle::dialect::PhiKernelOp>().op_name() ==
            "pd_op.shadow_feed") {
      auto out = op->result(0);
      if (out.use_count() == 1) {
        auto use_op = out.first_use().owner();
        auto op_info_parser = GetOpYamlInfoParser(use_op);
        for (size_t i = 0; i < use_op->num_operands(); ++i) {
          if (out == use_op->operand_source(i) &&
              op_info_parser->IsTensorAttribute(i)) {
            return true;
          }
        }
        return false;
      }
      return false;
    }
    return false;
  }

  bool Match(pir::Operation *op) const override {
    return IsSamePlaceShadowFeed(op) || IsTensorAttrShadowFeed(op);
  }

  void Rewrite(pir::Operation *op,
               pir::PatternRewriter &rewriter) const override {  // NOLINT
    auto in = op->operand_source(0);
    auto out = op->result(0);
    in.set_type(out.type());
    rewriter.ReplaceAllUsesWith(out, in);
    rewriter.EraseOp(op);
  }

 private:
  const pir::Block *block_;
  const phi::Place place_;
  const paddle::framework::Scope *scope_;
  std::unordered_map<::pir::Value, std::string> kwargs_map_;
};

class RemoveShadowFeedPass : public pir::PatternRewritePass {
 public:
  RemoveShadowFeedPass(const pir::Block *block,
                       const phi::Place &place,
                       const paddle::framework::Scope *scope)
      : pir::PatternRewritePass("remove_shadow_feed_pass", 0),
        block_(block),
        place_(place),
        scope_(scope) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add<RemoveShadowFeedPattern>(context, block_, place_, scope_);
    return ps;
  }

 private:
  const pir::Block *block_;
  const phi::Place place_;
  const paddle::framework::Scope *scope_;
};

}  // namespace

namespace pir {

std::unique_ptr<pir::Pass> CreateRemoveShadowFeedPass(
    const pir::Block *block,
    const phi::Place &place,
    const paddle::framework::Scope *scope) {
  return std::make_unique<RemoveShadowFeedPass>(block, place, scope);
}

}  // namespace pir

// REGISTER_IR_PASS(remove_shadow_feed_pass,
//                  RemoveShadowFeedPass);
