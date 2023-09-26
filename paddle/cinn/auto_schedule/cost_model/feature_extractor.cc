// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/auto_schedule/cost_model/feature_extractor.h"

#include <vector>

#include "paddle/cinn/common/target.h"
#include "paddle/cinn/common/type.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"
#include "paddle/cinn/ir/utils/ir_copy.h"
#include "paddle/cinn/optim/transform_polyfor_to_for.h"

namespace cinn {
namespace auto_schedule {

using namespace ::cinn::ir;  // NOLINT

FeatureExtractor::FeatureExtractor() {}

void FeatureExtractor::Visit(const Expr *x) {
  IRVisitorRequireReImpl::Visit(x);
}

Feature FeatureExtractor::Extract(const ir::ModuleExpr &mod_expr,
                                  const common::Target &target) {
  feature_ = Feature(target);
  for (const ir::Expr &e : mod_expr.GetExprs()) {
    Visit(&e);
  }
  return feature_;
}

#define VisitDoNothing(NodeType)                            \
  void FeatureExtractor::Visit(const NodeType *x) {         \
    std::vector<const Expr *> sub_exprs = x->expr_fields(); \
    for (const Expr *e : sub_exprs) {                       \
      if (e->defined()) {                                   \
        Visit(e);                                           \
      }                                                     \
    }                                                       \
  }

VisitDoNothing(IntImm);
VisitDoNothing(UIntImm);
VisitDoNothing(FloatImm);
VisitDoNothing(StringImm);

VisitDoNothing(Block);
VisitDoNothing(_Module_);
VisitDoNothing(_Var_);
VisitDoNothing(_LoweredFunc_);
VisitDoNothing(ScheduleBlock);
VisitDoNothing(ScheduleBlockRealize);
VisitDoNothing(Ramp);
VisitDoNothing(_Buffer_);
VisitDoNothing(_BufferRange_);

#define NotVisitExprFields(NodeType) \
  void FeatureExtractor::Visit(const NodeType *x) {}

NotVisitExprFields(_Tensor_)

#define VisitForDtypePattern(NodeType, member)                         \
  void FeatureExtractor::Visit(const NodeType *x) {                    \
    if (x->type() == common::F32() || x->type() == common::F16() ||    \
        x->type() == common::F64()) {                                  \
      feature_.CurrentLoopBlock().float_##member += x->type().lanes(); \
    } else {                                                           \
      feature_.CurrentLoopBlock().int_##member += x->type().lanes();   \
    }                                                                  \
    std::vector<const Expr *> sub_exprs = x->expr_fields();            \
    for (const Expr *e : sub_exprs) {                                  \
      if (e->defined()) {                                              \
        Visit(e);                                                      \
      }                                                                \
    }                                                                  \
  }

    VisitForDtypePattern(Add, add_or_sub);
VisitForDtypePattern(Sub, add_or_sub);
VisitForDtypePattern(Minus, add_or_sub);
VisitForDtypePattern(Mul, mul);
VisitForDtypePattern(Div, div_or_mod);
VisitForDtypePattern(Mod, div_or_mod);
VisitForDtypePattern(FracOp, div_or_mod);
VisitForDtypePattern(EQ, cmp);
VisitForDtypePattern(NE, cmp);
VisitForDtypePattern(GT, cmp);
VisitForDtypePattern(GE, cmp);
VisitForDtypePattern(LT, cmp);
VisitForDtypePattern(LE, cmp);
VisitForDtypePattern(Call, math_func);
VisitForDtypePattern(PrimitiveNode, math_func);
VisitForDtypePattern(Cast, other_call);
VisitForDtypePattern(Let, other_call);

#define VisitForMultiOperandsDtypePattern(NodeType, member)                   \
  void FeatureExtractor::Visit(const NodeType *x) {                           \
    if (x->type() == common::F32() || x->type() == common::F16() ||           \
        x->type() == common::F64()) {                                         \
      feature_.CurrentLoopBlock().float_##member +=                           \
          (x->operands().size() - 1);                                         \
    } else {                                                                  \
      feature_.CurrentLoopBlock().int_##member += (x->operands().size() - 1); \
    }                                                                         \
    std::vector<const Expr *> sub_exprs = x->expr_fields();                   \
    for (const Expr *e : sub_exprs) {                                         \
      if (e->defined()) {                                                     \
        Visit(e);                                                             \
      }                                                                       \
    }                                                                         \
  }

VisitForMultiOperandsDtypePattern(Sum, add_or_sub);
VisitForMultiOperandsDtypePattern(Product, mul);

#define VisitCountMemberPattern(NodeType, member)           \
  void FeatureExtractor::Visit(const NodeType *x) {         \
    feature_.CurrentLoopBlock().member += 1;                \
    std::vector<const Expr *> sub_exprs = x->expr_fields(); \
    for (const Expr *e : sub_exprs) {                       \
      if (e->defined()) {                                   \
        Visit(e);                                           \
      }                                                     \
    }                                                       \
  }

VisitCountMemberPattern(And, bool_op);
VisitCountMemberPattern(Or, bool_op);
VisitCountMemberPattern(Not, bool_op);
VisitCountMemberPattern(Max, select_op);
VisitCountMemberPattern(Min, select_op);
VisitCountMemberPattern(IfThenElse, select_op);
VisitCountMemberPattern(Select, select_op);
VisitCountMemberPattern(Alloc, mem_alloc);
VisitCountMemberPattern(Free, mem_free);
VisitCountMemberPattern(Load, mem_read);
VisitCountMemberPattern(Store, mem_write);

/* Visit for loops */

void FeatureExtractor::Visit(const For *x) {
  feature_.IntoLoopBlock();

  LoopBlockFeature &loop_feature = feature_.CurrentLoopBlock();
  if (x->min.is_constant() && x->extent.is_constant()) {
    loop_feature.loop_length =
        (x->extent.get_constant() - x->min.get_constant());
  } else {
    loop_feature.loop_length = -1;  // -1 represents unknown
  }

  if (x->is_parallel()) {
    loop_feature.loop_opt_type = ForOptimizeFeatureEnum::kParallel;
    loop_feature.len_vthread = loop_feature.loop_length;
  } else if (x->is_unrolled()) {
    loop_feature.loop_opt_type = ForOptimizeFeatureEnum::kUnroll;
  } else if (x->is_vectorized()) {
    loop_feature.loop_opt_type = ForOptimizeFeatureEnum::kVectorize;
    loop_feature.vectorize_factor = x->vectorize_info().factor;
  } else if (x->is_binded()) {
    loop_feature.loop_opt_type = ForOptimizeFeatureEnum::kGpuBind;
    const BindInfo &bind_info = x->bind_info();
    int offset = bind_info.offset;
    if (bind_info.for_type == ForType::GPUBlock) {
      if (offset == 0) {
        loop_feature.len_blockIdx_x = loop_feature.loop_length;
      } else if (offset == 1) {
        loop_feature.len_blockIdx_y = loop_feature.loop_length;
      } else if (offset == 2) {
        loop_feature.len_blockIdx_z = loop_feature.loop_length;
      }
    } else if (bind_info.for_type == ForType::GPUThread) {
      if (offset == 0) {
        loop_feature.len_threadIdx_x = loop_feature.loop_length;
      } else if (offset == 1) {
        loop_feature.len_threadIdx_y = loop_feature.loop_length;
      } else if (offset == 2) {
        loop_feature.len_threadIdx_z = loop_feature.loop_length;
      }
    }
  }

  std::vector<const Expr *> sub_exprs = x->expr_fields();
  for (const Expr *e : sub_exprs) {
    Visit(e);
  }

  feature_.ExitLoopBlock();
}

void FeatureExtractor::Visit(const PolyFor *x) {
  Expr copy = ir::ir_utils::IRCopy(Expr(x));
  feature_.IntoLoopBlock();
  optim::TransformPolyForToFor(&copy);
  ir::For *loop = copy.As<For>();
  CHECK(loop != nullptr);
  Visit(loop);
  feature_.ExitLoopBlock();
}

/* Visit for Reduce and Broadcast */

void FeatureExtractor::Visit(const Reduce *x) {
  if (x->type() == common::F32() || x->type() == common::F16() ||
      x->type() == common::F64()) {
    switch (x->reduce_type) {
      case Reduce::ReduceType::kSum:
        feature_.CurrentLoopBlock().float_reduce_sum_or_sub +=
            x->type().lanes();
        break;
      case Reduce::ReduceType::kSub:
        feature_.CurrentLoopBlock().float_reduce_sum_or_sub +=
            x->type().lanes();
        break;
      case Reduce::ReduceType::kDiv:
        feature_.CurrentLoopBlock().float_reduce_div += x->type().lanes();
        break;
      case Reduce::ReduceType::kMul:
        feature_.CurrentLoopBlock().float_reduce_mul += x->type().lanes();
        break;
      case Reduce::ReduceType::kMax:
        feature_.CurrentLoopBlock().float_reduce_max_or_min +=
            x->type().lanes();
        break;
      case Reduce::ReduceType::kMin:
        feature_.CurrentLoopBlock().float_reduce_max_or_min +=
            x->type().lanes();
        break;
    }
  } else {
    switch (x->reduce_type) {
      case Reduce::ReduceType::kSum:
        feature_.CurrentLoopBlock().int_reduce_sum_or_sub += x->type().lanes();
        break;
      case Reduce::ReduceType::kSub:
        feature_.CurrentLoopBlock().int_reduce_sum_or_sub += x->type().lanes();
        break;
      case Reduce::ReduceType::kDiv:
        feature_.CurrentLoopBlock().int_reduce_div += x->type().lanes();
        break;
      case Reduce::ReduceType::kMul:
        feature_.CurrentLoopBlock().int_reduce_mul += x->type().lanes();
        break;
      case Reduce::ReduceType::kMax:
        feature_.CurrentLoopBlock().int_reduce_max_or_min += x->type().lanes();
        break;
      case Reduce::ReduceType::kMin:
        feature_.CurrentLoopBlock().int_reduce_max_or_min += x->type().lanes();
        break;
    }
  }
  std::vector<const Expr *> sub_exprs = x->expr_fields();
  for (const Expr *e : sub_exprs) {
    Visit(e);
  }
}
VisitForDtypePattern(Broadcast, broadcast);

/* Visit for IntrinsicOp */
void FeatureExtractor::Visit(const IntrinsicOp *x) {
  switch (x->getKind()) {
#define __(op__)                                \
  case IntrinsicKind::k##op__:                  \
    Visit(llvm::dyn_cast<intrinsics::op__>(x)); \
    break;

    INTRINSIC_KIND_FOR_EACH(__)
#undef __
  }
}

VisitDoNothing(intrinsics::BufferGetDataHandle);
VisitDoNothing(intrinsics::BufferGetDataConstHandle);
VisitDoNothing(intrinsics::PodValueToX);
VisitDoNothing(intrinsics::BufferCreate);
VisitDoNothing(intrinsics::GetAddr);
VisitDoNothing(intrinsics::ArgsConstruct);

VisitForDtypePattern(intrinsics::BuiltinIntrin, other_call)

}  // namespace auto_schedule
}  // namespace cinn
