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

#include "paddle/cinn/hlir/framework/node.h"
#include "paddle/cinn/hlir/framework/op.h"
#include "paddle/cinn/hlir/framework/op_strategy.h"
#include "paddle/cinn/hlir/pe/broadcast.h"
#include "paddle/cinn/hlir/pe/ir_schedule_pe.h"
#include "paddle/cinn/hlir/pe/schedule.h"
#include "paddle/cinn/hlir/pe/transform.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"
#include "paddle/cinn/optim/ir_simplify.h"

namespace cinn {
namespace hlir {
namespace op {
using common::_CINNValuePack_;
using common::CINNValue;
using common::CINNValuePack;
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
    ReduceFunc cpu_reduce_func) {
  std::vector<int> reduce_axes;
  auto ndim = inputs[0]->shape.size();
  if (attrs.attr_store.count("dim")) {
    reduce_axes = absl::get<std::vector<int>>(attrs.attr_store.at("dim"));
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
    LOG(FATAL) << "reduce dimension is not set!";
  }

  bool keep_dim = false;
  if (attrs.attr_store.count("keep_dim")) {
    keep_dim = absl::get<bool>(attrs.attr_store.at("keep_dim"));
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

        if (target == common::DefaultNVGPUTarget()) {
          if (!WithoutLastDimInReduce(inputs[0]->shape, reduce_axes)) {
            VLOG(3) << "Do Two Step Block Reduce Compute!";
            auto res = gpu_reduce_with_last_axis_func(
                x, reduce_axes, keep_dim, tensor_name);
            auto stages = CreateStages(res);

            std::vector<CINNValue> cinn_values;
            for (auto &t : res) {
              cinn_values.emplace_back(t);
            }
            cinn_values.emplace_back(stages);
            *ret = CINNValuePack{cinn_values};
          } else {
            VLOG(3) << "Do Block Shuffle Reduce Compute!";
            auto res = gpu_reduce_without_last_axis_func(
                x, reduce_axes, keep_dim, tensor_name);
            auto stages = CreateStages(res);

            std::vector<CINNValue> cinn_values;
            for (auto &t : res) {
              cinn_values.emplace_back(t);
            }
            cinn_values.emplace_back(stages);
            *ret = CINNValuePack{cinn_values};
          }
        } else {
          VLOG(3) << "Do Reduce Compute!";
          auto out = cpu_reduce_func(x, reduce_axes, keep_dim, tensor_name);
          auto stages = CreateStages({out});

          std::vector<CINNValue> cinn_values{CINNValue(out), CINNValue(stages)};
          *ret = CINNValuePack{cinn_values};
        }
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
        // TODO(zhhsplendid): old reducetion schedule assumes all length-1
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
    if (target.arch == Target::Arch::NVGPU) {
      if (!WithoutLastDimInReduce(inputs[0]->shape, reduce_axes)) {
        if (arg_pack.size() == 4) {
          CHECK_EQ(vec_tensor.size(), 2);
          Expr out = vec_tensor[0];
          Expr tmp_out = vec_tensor[1];

          VLOG(3) << "Do IRCudaScheduleBlockReduceInternal Schedule!";
          pe::IRCudaScheduleBlockReduceInternal(
              ir_sch, tmp_out.as_tensor_ref(), out.as_tensor_ref(), target);

          std::vector<CINNValue> res{
              CINNValue(ir_sch.GetModule().GetExprs().at(0))};
          *ret = CINNValuePack{res};
        } else if (arg_pack.size() == 6) {
          CHECK_EQ(vec_tensor.size(), 3);
          Expr out = vec_tensor[0];
          Expr tmp_out = vec_tensor[1];
          Expr reduce_tmp_out = vec_tensor[2];

          VLOG(3) << "Do IRCudaScheduleBlockReduce Schedule!";
          pe::IRCudaScheduleBlockReduce(ir_sch,
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

          VLOG(3) << "Do IRCudaTwoStepReduceSchedule Schedule!";
          pe::IRCudaTwoStepReduceSchedule(ir_sch,
                                          reshape.as_tensor_ref(),
                                          reduce_tmp_out.as_tensor_ref(),
                                          tmp_out.as_tensor_ref(),
                                          out.as_tensor_ref(),
                                          common::DefaultNVGPUTarget());

          std::vector<CINNValue> res{
              CINNValue(ir_sch.GetModule().GetExprs().at(0))};
          *ret = CINNValuePack{res};
        } else if (arg_pack.size() == 5) {
          CHECK_EQ(vec_tensor.size(), 3);
          Expr out = vec_tensor[0];
          Expr tmp_out = vec_tensor[1];
          Expr reduce_tmp_out = vec_tensor[2];

          VLOG(3) << "Do IRCudaScheduleBlockReduce Schedule!";
          pe::IRCudaScheduleBlockReduce(ir_sch,
                                        reduce_tmp_out.as_tensor_ref(),
                                        tmp_out.as_tensor_ref(),
                                        out.as_tensor_ref(),
                                        common::DefaultNVGPUTarget());

          std::vector<CINNValue> res{
              CINNValue(ir_sch.GetModule().GetExprs().at(0))};
          *ret = CINNValuePack{res};
        } else {
          LOG(FATAL) << "Unkown Reduce Type!";
        }
      } else {
        if (arg_pack.size() == 2) {
          CHECK_EQ(vec_tensor.size(), 1);
          Expr reduce_out = vec_tensor[0];

          VLOG(3) << "Do IRCudaScheduleReduce Schedule!";
          pe::IRCudaScheduleReduce(
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

          VLOG(3) << "Do IRCudaScheduleBlockShuffleReduce Schedule!";
          pe::IRCudaScheduleBlockShuffleReduce(ir_sch,
                                               reduce_reshape.as_tensor_ref(),
                                               reduce_internal.as_tensor_ref(),
                                               reduce_out.as_tensor_ref(),
                                               target);

          std::vector<CINNValue> res{
              CINNValue(ir_sch.GetModule().GetExprs().at(0))};
          *ret = CINNValuePack{res};
        } else {
          LOG(FATAL) << "Unkown Reduce Type!";
        }
      }
    } else {
      std::vector<CINNValue> res{
          CINNValue(ir_sch.GetModule().GetExprs().at(0))};
      *ret = CINNValuePack{res};
    }
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(
      reduction_compute, reduction_schedule, "strategy." + op_name + ".x86", 1);

  return strategy;
}

#define STRATEGY_FOR_REDUCE(op_name_,                           \
                            reduce_op_,                         \
                            gpu_reduce_with_last_axis_func,     \
                            gpu_reduce_without_last_axis_func,  \
                            cpu_reduce_func)                    \
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
                             cpu_reduce_func);                  \
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

#undef STRATEGY_FOR_REDUCE

std::vector<shape_t> InferShapeForReduction(
    const std::vector<shape_t> &inputs_shape,
    const framework::AttrMapType &attrs) {
  CHECK(inputs_shape.size() == 1UL || inputs_shape.size() == 3UL);
  std::vector<int> dim;
  bool keep_dim = false;
  if (attrs.find("dim") != attrs.end()) {
    dim = absl::get<std::vector<int>>(attrs.at("dim"));
  }

  if (attrs.find("keep_dim") != attrs.end()) {
    keep_dim = absl::get<bool>(attrs.at("keep_dim"));
  }

  auto ndim = inputs_shape[0].size();
  CHECK_LE(dim.size(), ndim) << "reduce dim should no more than the input size";

  if (dim.empty()) {
    for (int i = 0; i < ndim; ++i) {
      dim.emplace_back(i);
    }
  } else {
    std::for_each(dim.begin(), dim.end(), [&ndim](int &x) {
      if (x < 0) x += ndim;
    });
  }

  std::vector<int> out_shapes;
  for (size_t i = 0; i < ndim; ++i) {
    if (std::find(dim.begin(), dim.end(), i) != dim.end()) {
      if (keep_dim) {
        out_shapes.push_back(1);
      }
    } else {
      out_shapes.push_back(inputs_shape[0][i]);
    }
  }

  if (out_shapes.empty()) {
    out_shapes.push_back(1);
  }

  VLOG(4) << "Reduce from input shape ["
          << cinn::utils::Join(inputs_shape[0], ",") << "] to output shape ["
          << cinn::utils::Join(out_shapes, ",") << "] with reduce dim ["
          << cinn::utils::Join(dim, ",") << "] and keep_dim is " << keep_dim;

  return {out_shapes};
}

std::vector<Type> InferDtypeForReduction(const std::vector<Type> &inputs_type,
                                         const framework::AttrMapType &attrs) {
  CHECK(!inputs_type.empty())
      << "The input's type size is 0! Please check again.";
  std::vector<Type> res{inputs_type[0]};
  return res;
}

std::vector<Type> InferDtypeForReductionBool(
    const std::vector<Type> &inputs_type, const framework::AttrMapType &attrs) {
  CHECK_EQ(inputs_type.size(), 1UL)
      << "The reduce should only has one input! Please check again.";
  CHECK(inputs_type[0].is_bool())
      << "The input's type should be bool! Please check.";
  return inputs_type;
}

std::vector<std::vector<std::string>> InferLayoutForReduction(
    const std::vector<framework::shape_t> &input_shapes,
    const std::vector<std::string> &input_layouts,
    const framework::NodeAttr &attrs,
    const Target &target) {
  CHECK_EQ(input_layouts.size(), 1U)
      << "The input's layouts size is not 1! Please check again.";
  std::vector<std::string> new_input_layouts = input_layouts;
  if (input_shapes[0].size() > 4) {
    // alter input layout back
    new_input_layouts[0] = "NCHW";
    VLOG(3) << "alter input layout from " << input_layouts[0] << " to "
            << new_input_layouts[0];
  }

  return {{""}, new_input_layouts};
}

std::vector<shape_t> InferShapeForBnOptimize(
    const std::vector<shape_t> &inputs_shape,
    const framework::AttrMapType &attrs) {
  auto shapes = InferShapeForReduction(inputs_shape, attrs);
  CHECK_GE(shapes.size(), 1) << "shapes's size less than 1, please check!";
  return {shapes[0], shapes[0]};
}

std::vector<Type> InferDtypeForBnOptimize(const std::vector<Type> &inputs_type,
                                          const framework::AttrMapType &attrs) {
  CHECK(!inputs_type.empty())
      << "The input's type size is 0! Please check again.";
  return {inputs_type[0], inputs_type[0]};
}

std::vector<std::vector<std::string>> InferLayoutForBnOptimize(
    const std::vector<framework::shape_t> &input_shapes,
    const std::vector<std::string> &input_layouts,
    const framework::NodeAttr &attrs,
    const Target &target) {
  return {{"", ""}, {"", ""}};
}

}  // namespace op
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(reduce_ops) {
#define CINN_REGISTER_REDUCTION_WITH_DTYPE(op__, op_stragegy__, dtype__)   \
  CINN_REGISTER_OP(op__)                                                   \
      .describe(#op__ " function")                                         \
      .set_num_inputs(1)                                                   \
      .set_num_outputs(1)                                                  \
      .set_attr<cinn::hlir::framework::StrategyFunction>(                  \
          "CINNStrategy", cinn::hlir::op::StrategyFor##op_stragegy__)      \
      .set_attr("infershape",                                              \
                MakeOpFunction(cinn::hlir::op::InferShapeForReduction))    \
      .set_attr(                                                           \
          "inferdtype",                                                    \
          MakeOpFunction(cinn::hlir::op::InferDtypeForReduction##dtype__)) \
      .set_attr("inferlayout",                                             \
                MakeOpFunction(cinn::hlir::op::InferLayoutForReduction))   \
      .set_attr<cinn::hlir::framework::OpPatternKind>(                     \
          "OpPattern", cinn::hlir::framework::OpPatternKind::kReduction)   \
      .set_support_level(4);

#define CINN_REGISTER_REDUCTION(op__, op_stragegy__) \
  CINN_REGISTER_REDUCTION_WITH_DTYPE(op__, op_stragegy__, )

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
