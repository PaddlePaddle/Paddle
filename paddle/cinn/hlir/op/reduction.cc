// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/hlir/pe/reduction.h"

#include <algorithm>
#include <iostream>
#include <vector>

#include "paddle/cinn/adt/op_equation_context.h"
#include "paddle/cinn/hlir/framework/op.h"
#include "paddle/cinn/hlir/framework/op_strategy.h"
#include "paddle/cinn/hlir/pe/broadcast.h"
#include "paddle/cinn/hlir/pe/ir_schedule_pe.h"
#include "paddle/cinn/hlir/pe/schedule.h"
#include "paddle/cinn/hlir/pe/transform.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"
#include "paddle/cinn/optim/ir_simplify.h"
#include "paddle/cinn/runtime/flags.h"

PD_DECLARE_bool(cinn_enable_map_expr);

PD_DECLARE_bool(cinn_new_group_scheduler);

PD_DECLARE_bool(cinn_bucket_compile);

namespace cinn {
namespace hlir {
namespace op {
using cinn::common::_CINNValuePack_;
using cinn::common::CINNValue;
using cinn::common::CINNValuePack;
using framework::OpStrategy;
using framework::shape_t;
using framework::StrategyFunction;

using BlockReduceFunc =
    std::function<std::vector<ir::Tensor>(const ir::Tensor &,
                                          const std::vector<int> &,
                                          const bool,
                                          const std::string &)>;
using ReduceFunc = std::function<ir::Tensor(const ir::Tensor &,
                                            const std::vector<int> &,
                                            const bool,
                                            const std::string &)>;

std::shared_ptr<OpStrategy> StrategyForReduce(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<int>> &output_shapes,
    const Target &target,
    const std::string &op_name,
    BlockReduceFunc gpu_reduce_with_last_axis_func,
    BlockReduceFunc gpu_reduce_without_last_axis_func,
    ReduceFunc common_reduce_func) {
  std::vector<int> reduce_axes;
  auto ndim = inputs[0]->shape.size();
  if (attrs.attr_store.count("axis")) {
    reduce_axes = [&] {
      if (absl::holds_alternative<std::vector<int64_t>>(
              attrs.attr_store.at("axis"))) {
        const auto &dim_attr =
            absl::get<std::vector<int64_t>>(attrs.attr_store.at("axis"));
        return std::vector<int>(dim_attr.begin(), dim_attr.end());
      } else if (absl::holds_alternative<std::vector<int>>(
                     attrs.attr_store.at("axis"))) {
        return absl::get<std::vector<int>>(attrs.attr_store.at("axis"));
      } else if (absl::holds_alternative<bool>(attrs.attr_store.at("axis"))) {
        return std::vector<int>{};
      } else {
        PADDLE_THROW(phi::errors::InvalidArgument(
            "reduce dimension's type is invalid!"));
      }
    }();
    if (reduce_axes.empty()) {
      for (int i = 0; i < ndim; ++i) {
        reduce_axes.push_back(i);
      }
    } else {
      std::for_each(reduce_axes.begin(), reduce_axes.end(), [&ndim](int &x) {
        if (x < 0) x += ndim;
      });
    }
    std::sort(reduce_axes.begin(), reduce_axes.end());
    // check reduce_axes
    CHECK_LE(reduce_axes.size(), ndim);
    CHECK_LT(reduce_axes.back(), ndim);
    for (int idx = 1; idx < reduce_axes.size(); ++idx) {
      CHECK_NE(reduce_axes[idx - 1], reduce_axes[idx]);
    }
  } else {
    PADDLE_THROW(phi::errors::InvalidArgument("reduce dimension is not set!"));
  }

  bool keepdim = false;
  if (attrs.attr_store.count("keepdim")) {
    keepdim = absl::get<bool>(attrs.attr_store.at("keepdim"));
  }

  auto WithoutLastDimInReduce = [](const std::vector<ir::Expr> &inshape,
                                   const std::vector<int> &axes) {
    // if last axis is in reduce.
    if (std::find(axes.begin(), axes.end(), inshape.size() - 1) != axes.end() ||
        std::find(axes.begin(), axes.end(), -1) != axes.end()) {
      return false;
    }

    int sum_last_axes = 1;
    for (int idx = axes.back() + 1; idx < inshape.size(); ++idx) {
      sum_last_axes *= inshape[idx].as_int32();
    }

    if (sum_last_axes > 1) {
      return true;
    } else {
      return false;
    }
  };

  framework::CINNCompute reduction_compute([=](lang::Args args,
                                               lang::RetValue *ret) {
    PADDLE_ENFORCE_EQ(
        !args.empty(),
        true,
        phi::errors::InvalidArgument(
            "The input argument of %s compute is empty! Please check.",
            op_name));
    CINNValuePack arg_packs = args[0];
    PADDLE_ENFORCE_EQ(
        arg_packs.size(),
        2U,
        phi::errors::InvalidArgument(
            "There should be 2 input args for %s compute", op_name));
    PADDLE_ENFORCE_EQ(arg_packs[1].is_string(),
                      true,
                      phi::errors::InvalidArgument(
                          "The arg_packs[1] is not empty! Please check."));
    std::string tensor_name = arg_packs[1].operator std::string();
    Expr x_expr = arg_packs[0];
    PADDLE_ENFORCE_NOT_NULL(x_expr.as_tensor(),
                            phi::errors::InvalidArgument(
                                "The x_expr can not as tensor! Please check."));
    ir::Tensor x = x_expr.as_tensor_ref();

    std::unordered_set<std::string> bool_reduce_op = {"reduce_all",
                                                      "reduce_any"};
    PADDLE_ENFORCE_EQ(!bool_reduce_op.count(op_name) || x->type().is_bool(),
                      true,
                      phi::errors::InvalidArgument(
                          "The type of input argument %s of %s should be bool, "
                          "but get %s! Please check.",
                          x->name,
                          op_name,
                          x->type().to_string()));

    const auto &NaiveCompute = [&]() {
      VLOG(3) << "Do Reduce Compute!";
      auto out = common_reduce_func(x, reduce_axes, keepdim, tensor_name);

      std::vector<CINNValue> cinn_values{CINNValue(out)};
      *ret = CINNValuePack{cinn_values};
    };
    auto reductionComputeNvHygon = [&] {
      if (!FLAGS_cinn_enable_map_expr && !FLAGS_cinn_new_group_scheduler) {
        if (!WithoutLastDimInReduce(inputs[0]->shape, reduce_axes)) {
          VLOG(3) << "Do Two Step Block Reduce Compute!";
          auto res = gpu_reduce_with_last_axis_func(
              x, reduce_axes, keepdim, tensor_name);

          std::vector<CINNValue> cinn_values;
          for (auto &t : res) {
            cinn_values.emplace_back(t);
          }
          *ret = CINNValuePack{cinn_values};
        } else {
          VLOG(3) << "Do Block Shuffle Reduce Compute!";
          auto res = gpu_reduce_without_last_axis_func(
              x, reduce_axes, keepdim, tensor_name);

          std::vector<CINNValue> cinn_values;
          for (auto &t : res) {
            cinn_values.emplace_back(t);
          }
          *ret = CINNValuePack{cinn_values};
        }
      } else {
        NaiveCompute();
      }
    };
    target.arch.Match(
        [&](common::NVGPUArch) { reductionComputeNvHygon(); },
        [&](std::variant<common::UnknownArch,
                         common::X86Arch,
                         common::ARMArch>) { NaiveCompute(); },
        [&](common::HygonDCUArchHIP) { reductionComputeNvHygon(); });
  });

  framework::CINNSchedule reduction_schedule([=](lang::Args args,
                                                 lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input argument of " << op_name
                         << " schedule is empty! Please check.";

    CINNValuePack arg_pack = args[0];
    CHECK_GE(arg_pack.size(), 2UL);
    CHECK_LE(arg_pack.size(), 8UL);
    std::vector<Expr> vec_ast;
    std::vector<Expr> vec_tensor;
    for (int i = 0; i < arg_pack.size(); i++) {
      if (arg_pack[i].is_expr()) {
        Expr temp = arg_pack[i];
        // TODO(zhhsplendid): old reduction schedule assumes all length-1
        // for loops are simplified, but it is not after we add length-1
        // back. Reduction schedule is complex and we haven't changed it to
        // support the length-1 for loop yet. So we simplify here. The todo
        // is that remove SimplifyForLoops below and change reduction schedule
        optim::SimplifyForLoops(&temp);
        optim::SimplifyBlocks(&temp);
        vec_ast.emplace_back(temp);
      } else if (arg_pack[i].is_tensor()) {
        Expr temp = arg_pack[i];
        vec_tensor.emplace_back(temp);
      }
    }
    CHECK(!vec_ast.empty());
    ir::ModuleExpr mod_expr(vec_ast);
    ir::IRSchedule ir_sch(mod_expr);
    ir_sch.MergeExprs();
    const auto ReduceSchedule = [&]() {
      if (!WithoutLastDimInReduce(inputs[0]->shape, reduce_axes)) {
        if (arg_pack.size() == 4) {
          CHECK_EQ(vec_tensor.size(), 2);
          Expr out = vec_tensor[0];
          Expr tmp_out = vec_tensor[1];

          VLOG(3) << "Do IRGpuScheduleBlockReduceInternal Schedule!";
          pe::IRGpuScheduleBlockReduceInternal(
              ir_sch, tmp_out.as_tensor_ref(), out.as_tensor_ref(), target);

          std::vector<CINNValue> res{
              CINNValue(ir_sch.GetModule().GetExprs().at(0))};
          *ret = CINNValuePack{res};
        } else if (arg_pack.size() == 6) {
          CHECK_EQ(vec_tensor.size(), 3);
          Expr out = vec_tensor[0];
          Expr tmp_out = vec_tensor[1];
          Expr reduce_tmp_out = vec_tensor[2];

          VLOG(3) << "Do IRGpuScheduleBlockReduce Schedule!";
          pe::IRGpuScheduleBlockReduce(ir_sch,
                                       reduce_tmp_out.as_tensor_ref(),
                                       tmp_out.as_tensor_ref(),
                                       out.as_tensor_ref(),
                                       target);

          std::vector<CINNValue> res{
              CINNValue(ir_sch.GetModule().GetExprs().at(0))};
          *ret = CINNValuePack{res};
        } else if (arg_pack.size() == 7) {
          CHECK_EQ(vec_tensor.size(), 4);
          Expr out = vec_tensor[0];
          Expr tmp_out = vec_tensor[1];
          Expr reduce_tmp_out = vec_tensor[2];
          Expr reshape = vec_tensor[3];

          VLOG(3) << "Do IRGpuTwoStepReduceSchedule Schedule!";
          pe::IRGpuTwoStepReduceSchedule(ir_sch,
                                         reshape.as_tensor_ref(),
                                         reduce_tmp_out.as_tensor_ref(),
                                         tmp_out.as_tensor_ref(),
                                         out.as_tensor_ref(),
                                         cinn::common::DefaultDeviceTarget());

          std::vector<CINNValue> res{
              CINNValue(ir_sch.GetModule().GetExprs().at(0))};
          *ret = CINNValuePack{res};
        } else if (arg_pack.size() == 5) {
          CHECK_EQ(vec_tensor.size(), 3);
          Expr out = vec_tensor[0];
          Expr tmp_out = vec_tensor[1];
          Expr reduce_tmp_out = vec_tensor[2];

          VLOG(3) << "Do IRGpuScheduleBlockReduce Schedule!";
          pe::IRGpuScheduleBlockReduce(ir_sch,
                                       reduce_tmp_out.as_tensor_ref(),
                                       tmp_out.as_tensor_ref(),
                                       out.as_tensor_ref(),
                                       cinn::common::DefaultDeviceTarget());

          std::vector<CINNValue> res{
              CINNValue(ir_sch.GetModule().GetExprs().at(0))};
          *ret = CINNValuePack{res};
        } else {
          PADDLE_THROW(phi::errors::InvalidArgument("Unkown Reduce Type!"));
        }
      } else {
        if (arg_pack.size() == 2) {
          CHECK_EQ(vec_tensor.size(), 1);
          Expr reduce_out = vec_tensor[0];

          VLOG(3) << "Do IRGpuScheduleReduce Schedule!";
          pe::IRGpuScheduleReduce(
              ir_sch,
              reduce_out.as_tensor_ref(),
              inputs[0]->shape.size() - reduce_axes.back() - 1,
              target);

          std::vector<CINNValue> res{
              CINNValue(ir_sch.GetModule().GetExprs().at(0))};
          *ret = CINNValuePack{res};
        } else if (arg_pack.size() == 6) {
          CHECK_EQ(vec_tensor.size(), 3);
          Expr reduce_out = vec_tensor[0];
          Expr reduce_internal = vec_tensor[1];
          Expr reduce_reshape = vec_tensor[2];

          VLOG(3) << "Do IRGpuScheduleBlockShuffleReduce Schedule!";
          pe::IRGpuScheduleBlockShuffleReduce(ir_sch,
                                              reduce_reshape.as_tensor_ref(),
                                              reduce_internal.as_tensor_ref(),
                                              reduce_out.as_tensor_ref(),
                                              target);

          std::vector<CINNValue> res{
              CINNValue(ir_sch.GetModule().GetExprs().at(0))};
          *ret = CINNValuePack{res};
        } else {
          PADDLE_THROW(phi::errors::InvalidArgument("Unkown Reduce Type!"));
        }
      }
    };
    target.arch.Match([&](common::UnknownArch) { CINN_NOT_IMPLEMENTED; },
                      [&](common::X86Arch) {
                        std::vector<CINNValue> res{
                            CINNValue(ir_sch.GetModule().GetExprs().at(0))};
                        *ret = CINNValuePack{res};
                      },
                      [&](common::ARMArch) {
                        std::vector<CINNValue> res{
                            CINNValue(ir_sch.GetModule().GetExprs().at(0))};
                        *ret = CINNValuePack{res};
                      },
                      [&](common::NVGPUArch) {
                        if (!FLAGS_cinn_new_group_scheduler) {
                          ReduceSchedule();
                        } else {
                          std::vector<CINNValue> res{
                              CINNValue(ir_sch.GetModule().GetExprs().at(0))};
                          *ret = CINNValuePack{res};
                        }
                      },
                      [&](common::HygonDCUArchHIP) {
                        if (!FLAGS_cinn_new_group_scheduler) {
                          ReduceSchedule();
                        } else {
                          std::vector<CINNValue> res{
                              CINNValue(ir_sch.GetModule().GetExprs().at(0))};
                          *ret = CINNValuePack{res};
                        }
                      });
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(
      reduction_compute, reduction_schedule, "strategy." + op_name + ".x86", 1);

  return strategy;
}

std::shared_ptr<OpStrategy> StrategyForReduceSymbolic(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<ir::Dim>> &output_shapes,
    const Target &target,
    const std::string &op_name,
    BlockReduceFunc gpu_reduce_with_last_axis_func,
    BlockReduceFunc gpu_reduce_without_last_axis_func,
    ReduceFunc common_reduce_func) {
  std::vector<int> reduce_axes;
  auto ndim = inputs[0]->shape.size();
  if (attrs.attr_store.count("axis")) {
    reduce_axes = [&] {
      if (absl::holds_alternative<std::vector<int64_t>>(
              attrs.attr_store.at("axis"))) {
        const auto &dim_attr =
            absl::get<std::vector<int64_t>>(attrs.attr_store.at("axis"));
        return std::vector<int>(dim_attr.begin(), dim_attr.end());
      } else if (absl::holds_alternative<std::vector<int>>(
                     attrs.attr_store.at("axis"))) {
        return absl::get<std::vector<int>>(attrs.attr_store.at("axis"));
      } else if (absl::holds_alternative<bool>(attrs.attr_store.at("axis"))) {
        return std::vector<int>{};
      } else {
        PADDLE_THROW(phi::errors::InvalidArgument(
            "reduce dimension's type is invalid!"));
      }
    }();
    if (reduce_axes.empty()) {
      for (int i = 0; i < ndim; ++i) {
        reduce_axes.push_back(i);
      }
    } else {
      std::for_each(reduce_axes.begin(), reduce_axes.end(), [&ndim](int &x) {
        if (x < 0) x += ndim;
      });
    }
    std::sort(reduce_axes.begin(), reduce_axes.end());
    CHECK_LE(reduce_axes.size(), ndim);
    CHECK_LT(reduce_axes.back(), ndim);
    for (int idx = 1; idx < reduce_axes.size(); ++idx) {
      CHECK_NE(reduce_axes[idx - 1], reduce_axes[idx]);
    }
  } else {
    PADDLE_THROW(phi::errors::InvalidArgument("reduce dimension is not set!"));
  }

  bool keepdim = false;
  if (attrs.attr_store.count("keepdim")) {
    keepdim = absl::get<bool>(attrs.attr_store.at("keepdim"));
  }

  framework::CINNCompute reduction_compute(
      [=](lang::Args args, lang::RetValue *ret) {
        CHECK(!args.empty()) << "The input argument of " << op_name
                             << " compute is empty! Please check.";
        CINNValuePack arg_packs = args[0];
        CHECK_EQ(arg_packs.size(), 2U)
            << "There should be 2 input args for " << op_name << " compute";
        CHECK(arg_packs[1].is_string());
        std::string tensor_name = arg_packs[1].operator std::string();
        Expr x_expr = arg_packs[0];
        CHECK(x_expr.as_tensor());
        ir::Tensor x = x_expr.as_tensor_ref();

        std::unordered_set<std::string> bool_reduce_op = {"reduce_all",
                                                          "reduce_any"};
        CHECK(!bool_reduce_op.count(op_name) || x->type().is_bool())
            << "The type of input argument " << x->name << " of " << op_name
            << " should be bool, but get " << x->type() << "! Please check.";

        VLOG(3) << "Do Reduce Compute!";
        auto out = common_reduce_func(x, reduce_axes, keepdim, tensor_name);

        std::vector<CINNValue> cinn_values{CINNValue(out)};
        *ret = CINNValuePack{cinn_values};
      });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(
      reduction_compute, lang::PackedFunc(), "strategy." + op_name + ".x86", 1);

  return strategy;
}

#define STRATEGY_FOR_REDUCE(op_name_,                           \
                            reduce_op_,                         \
                            gpu_reduce_with_last_axis_func,     \
                            gpu_reduce_without_last_axis_func,  \
                            common_reduce_func)                 \
  std::shared_ptr<OpStrategy> StrategyFor##reduce_op_(          \
      const framework::NodeAttr &attrs,                         \
      const std::vector<ir::Tensor> &inputs,                    \
      const std::vector<Type> &out_type,                        \
      const std::vector<std::vector<int>> &output_shapes,       \
      const Target &target) {                                   \
    return StrategyForReduce(attrs,                             \
                             inputs,                            \
                             out_type,                          \
                             output_shapes,                     \
                             target,                            \
                             #op_name_,                         \
                             gpu_reduce_with_last_axis_func,    \
                             gpu_reduce_without_last_axis_func, \
                             common_reduce_func);               \
  }

#define STRATEGY_FOR_REDUCE_SYMBOLIC(op_name_,                          \
                                     reduce_op_,                        \
                                     gpu_reduce_with_last_axis_func,    \
                                     gpu_reduce_without_last_axis_func, \
                                     common_reduce_func)                \
  std::shared_ptr<OpStrategy> StrategyFor##reduce_op_##Symbolic(        \
      const framework::NodeAttr &attrs,                                 \
      const std::vector<ir::Tensor> &inputs,                            \
      const std::vector<Type> &out_type,                                \
      const std::vector<std::vector<ir::Dim>> &output_shapes,           \
      const Target &target) {                                           \
    return StrategyForReduceSymbolic(attrs,                             \
                                     inputs,                            \
                                     out_type,                          \
                                     output_shapes,                     \
                                     target,                            \
                                     #op_name_,                         \
                                     gpu_reduce_with_last_axis_func,    \
                                     gpu_reduce_without_last_axis_func, \
                                     common_reduce_func);               \
  }

STRATEGY_FOR_REDUCE(reduce_sum,
                    ReduceSum,
                    pe::TwoStepBlockReduceSum,
                    pe::BlockShuffleReduceSum,
                    pe::ReduceSum);
STRATEGY_FOR_REDUCE(reduce_prod,
                    ReduceProd,
                    pe::TwoStepBlockReduceProd,
                    pe::BlockShuffleReduceProd,
                    pe::ReduceProd);
STRATEGY_FOR_REDUCE(reduce_max,
                    ReduceMax,
                    pe::TwoStepBlockReduceMax,
                    pe::BlockShuffleReduceMax,
                    pe::ReduceMax);
STRATEGY_FOR_REDUCE(reduce_min,
                    ReduceMin,
                    pe::TwoStepBlockReduceMin,
                    pe::BlockShuffleReduceMin,
                    pe::ReduceMin);
STRATEGY_FOR_REDUCE(reduce_all,
                    ReduceAll,
                    pe::TwoStepBlockReduceAll,
                    pe::BlockShuffleReduceAll,
                    pe::ReduceAll);
STRATEGY_FOR_REDUCE(reduce_any,
                    ReduceAny,
                    pe::TwoStepBlockReduceAny,
                    pe::BlockShuffleReduceAny,
                    pe::ReduceAny);

STRATEGY_FOR_REDUCE_SYMBOLIC(reduce_sum,
                             ReduceSum,
                             pe::TwoStepBlockReduceSum,
                             pe::BlockShuffleReduceSum,
                             pe::ReduceSum);
STRATEGY_FOR_REDUCE_SYMBOLIC(reduce_prod,
                             ReduceProd,
                             pe::TwoStepBlockReduceProd,
                             pe::BlockShuffleReduceProd,
                             pe::ReduceProd);
STRATEGY_FOR_REDUCE_SYMBOLIC(reduce_max,
                             ReduceMax,
                             pe::TwoStepBlockReduceMax,
                             pe::BlockShuffleReduceMax,
                             pe::ReduceMax);
STRATEGY_FOR_REDUCE_SYMBOLIC(reduce_min,
                             ReduceMin,
                             pe::TwoStepBlockReduceMin,
                             pe::BlockShuffleReduceMin,
                             pe::ReduceMin);
STRATEGY_FOR_REDUCE_SYMBOLIC(reduce_all,
                             ReduceAll,
                             pe::TwoStepBlockReduceAll,
                             pe::BlockShuffleReduceAll,
                             pe::ReduceAll);
STRATEGY_FOR_REDUCE_SYMBOLIC(reduce_any,
                             ReduceAny,
                             pe::TwoStepBlockReduceAny,
                             pe::BlockShuffleReduceAny,
                             pe::ReduceAny);

#undef STRATEGY_FOR_REDUCE
#undef STRATEGY_FOR_REDUCE_SYMBOLIC

}  // namespace op
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(reduce_ops) {
#define CINN_REGISTER_REDUCTION_WITH_DTYPE(op__, op_strategy__, dtype__) \
  CINN_REGISTER_OP(op__)                                                 \
      .describe(#op__ " function")                                       \
      .set_num_inputs(1)                                                 \
      .set_num_outputs(1)                                                \
      .set_attr<cinn::hlir::framework::StrategyFunction>(                \
          "CINNStrategy", cinn::hlir::op::StrategyFor##op_strategy__)    \
      .set_attr<cinn::hlir::framework::StrategyFunctionSymbolic>(        \
          "CINNStrategySymbolic",                                        \
          cinn::hlir::op::StrategyFor##op_strategy__##Symbolic)          \
      .set_attr<cinn::hlir::framework::OpPatternKind>(                   \
          "OpPattern", cinn::hlir::framework::OpPatternKind::kReduction) \
      .set_support_level(4);

#define CINN_REGISTER_REDUCTION(op__, op_strategy__) \
  CINN_REGISTER_REDUCTION_WITH_DTYPE(op__, op_strategy__, )

  CINN_REGISTER_REDUCTION(reduce_sum, ReduceSum);
  CINN_REGISTER_REDUCTION(reduce_prod, ReduceProd);
  CINN_REGISTER_REDUCTION(reduce_max, ReduceMax);
  CINN_REGISTER_REDUCTION(reduce_min, ReduceMin);

#undef CINN_REGISTER_REDUCTION

  CINN_REGISTER_REDUCTION_WITH_DTYPE(reduce_all, ReduceAll, Bool);
  CINN_REGISTER_REDUCTION_WITH_DTYPE(reduce_any, ReduceAny, Bool);

#undef CINN_REGISTER_REDUCTION_WITH_DTYPE

  return true;
}
