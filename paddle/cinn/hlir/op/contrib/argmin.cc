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

#include "paddle/cinn/hlir/op/contrib/argmin.h"

#include <iostream>
#include <vector>

#include "paddle/cinn/common/cas.h"
#include "paddle/cinn/common/cinn_value.h"
#include "paddle/cinn/common/common.h"
#include "paddle/cinn/common/context.h"
#include "paddle/cinn/common/macros.h"
#include "paddle/cinn/hlir/framework/node.h"
#include "paddle/cinn/hlir/framework/op.h"
#include "paddle/cinn/hlir/framework/op_strategy.h"
#include "paddle/cinn/hlir/op/contrib/sort.h"
#include "paddle/cinn/hlir/pe/ir_schedule_pe.h"
#include "paddle/cinn/hlir/pe/nn.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"
#include "paddle/cinn/ir/tensor.h"
#include "paddle/cinn/lang/builtin.h"
#include "paddle/cinn/lang/compute.h"

namespace cinn {
namespace hlir {
namespace op {

using common::CINNValue;
using framework::shape_t;
using ir::Tensor;

std::vector<Tensor> Argmin(const Tensor &in_tensor,
                           const common::Target &target,
                           poly::StageMap stages,
                           const int &axis,
                           const bool &keep_dims,
                           const std::string &name) {
  auto shape = in_tensor->shape;
  auto ndim = shape.size();
  CHECK_GT(ndim, 0) << "tensor's dim must be more than 0";

  int pos_axis = axis;
  if (axis < 0) {
    pos_axis = static_cast<int>(ndim) + axis;
  }
  CHECK_LT(pos_axis, ndim) << "Axis must be less than tensor's dim";
  CHECK_GE(pos_axis, 0) << "Axis must be more than 0";

  std::vector<Expr> output_shape;
  for (int i = 0; i < shape.size(); ++i) {
    CHECK(shape[i].is_constant())
        << "Input tensor's shape should be constant value.";
    if (pos_axis == i) {
      if (keep_dims) {
        output_shape.push_back(Expr(1));
      }
    } else {
      output_shape.push_back(shape[i]);
    }
  }
  if (output_shape.empty()) {
    output_shape.push_back(Expr(1));
  }
  auto sort_index =
      ArgSort(in_tensor, target, stages, pos_axis, true, name + "_index");
  auto res = Compute(
      output_shape,
      [=](const std::vector<Expr> &indices) {
        std::vector<Expr> eval_indices(indices);
        if (!keep_dims && ndim > 1) {
          eval_indices.insert(eval_indices.begin() + pos_axis, Expr(0));
        } else {
          eval_indices[pos_axis] = Expr(0);
        }
        return sort_index.at(0)(eval_indices);
      },
      name);
  stages->InsertLazily(sort_index.at(0));
  return {res, sort_index.at(0), sort_index.at(1)};
}

std::shared_ptr<framework::OpStrategy> StrategyForArgmin(
    const framework::NodeAttr &attrs,
    const std::vector<Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<int>> &output_shapes,
    const Target &target) {
  int axis;
  bool keep_dims = false;

  if (attrs.attr_store.count("axis")) {
    axis = absl::get<int>(attrs.attr_store.at("axis"));
  } else {
    LOG(FATAL) << "reduce dimension is not set!";
  }
  if (attrs.attr_store.count("keep_dim")) {
    keep_dims = absl::get<bool>(attrs.attr_store.at("keep_dim"));
  }

  framework::CINNCompute argmin_compute(
      [=](lang::Args args, lang::RetValue *ret) {
        CHECK(!args.empty())
            << "The input argument of argmin compute is empty! Please check.";
        common::CINNValuePack pack_args = args[0];
        CHECK_GE(pack_args.size(), 1U)
            << "There should be 1 input args for argmax compute";
        Expr in_expr = pack_args[0];
        CHECK(in_expr.as_tensor());
        Tensor in_tensor = in_expr.as_tensor_ref();
        auto stages = CreateStages({in_tensor});
        CHECK_EQ(pack_args.size(), 2U);
        CHECK(pack_args[1].is_string());
        std::string tensor_name = pack_args[1].operator std::string();
        auto out_tensor =
            Argmin(in_tensor, target, stages, axis, keep_dims, tensor_name);

        stages->InsertLazily(out_tensor[0]);
        std::vector<CINNValue> cinn_values{CINNValue(out_tensor[0]),
                                           CINNValue(out_tensor[1]),
                                           CINNValue(out_tensor[2]),
                                           CINNValue(stages)};
        *ret = common::CINNValuePack{cinn_values};
      });

  framework::CINNSchedule argmin_schedule([=](lang::Args args,
                                              lang::RetValue *ret) {
    CHECK(!args.empty())
        << "The input argument of arange_schedule is empty! Please check.\n";
    common::CINNValuePack arg_pack = args[0];
    std::vector<Expr> vec_ast;
    for (int i = 0; i < arg_pack.size(); i++) {
      if (arg_pack[i].is_expr()) {
        Expr temp = arg_pack[i];
        vec_ast.emplace_back(temp);
      }
    }
    CHECK(!vec_ast.empty());
    ir::ModuleExpr mod_expr(vec_ast);
    ir::IRSchedule ir_sch(mod_expr);
    ir_sch.MergeExprs();
    auto blocks = ir_sch.GetAllBlocks();
    // TODO(zhhsplendid): It needs to be rewritten according to the
    // reduction_min operator to improve performance. Do not use local
    // variables, because the size will exceed the limit.
    ir_sch.SetBuffer(blocks[0], "local");
    ir_sch.SetBuffer(blocks[1], "local");
    int64_t prod_size = std::accumulate(output_shapes[0].begin(),
                                        output_shapes[0].end(),
                                        1,
                                        std::multiplies<int>());
    if (prod_size > 1 && target.arch == Target::Arch::X86) {
      pe::IRScheduleInjectiveCPU(ir_sch, output_shapes.front(), target, true);
    }
    std::vector<common::CINNValue> res{
        common::CINNValue(ir_sch.GetModule().GetExprs().at(0))};
    *ret = common::CINNValuePack{res};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(argmin_compute, argmin_schedule, "strategy.argmin.x86", 1);

  return strategy;
}

std::vector<shape_t> InferShapeForArgmin(
    const std::vector<shape_t> &inputs_shape,
    const framework::AttrMapType &attrs) {
  CHECK_EQ(inputs_shape.size(), 1UL);
  auto ndim = inputs_shape[0].size();
  int axis;
  bool keep_dim;

  CHECK(attrs.find("axis") != attrs.end());
  axis = absl::get<int>(attrs.at("axis"));
  if (ndim > 0) {
    if (axis < 0) {
      axis = static_cast<int>(ndim) + axis;
    }
    CHECK_LT(axis, ndim) << "Axis must be less than tensor's dim";
    CHECK_GE(axis, 0) << "Axis must be more than 0";
  } else {
    // 0D Tensor
    CHECK(axis == 0 || axis == -1)
        << "Axis must be 0 or -1 if input tensor is 0-dim";
  }

  CHECK(attrs.find("keep_dim") != attrs.end());
  keep_dim = absl::get<bool>(attrs.at("keep_dim"));

  std::vector<int> out_shapes;
  for (size_t i = 0; i < ndim; ++i) {
    if (axis == i) {
      if (keep_dim) {
        out_shapes.push_back(1);
      }
    } else {
      out_shapes.push_back(inputs_shape[0][i]);
    }
  }

  if (keep_dim) {
    CHECK_EQ(ndim, out_shapes.size());
  } else {
    CHECK(ndim - 1 == out_shapes.size() || ndim == 0 && out_shapes.empty());
  }

  return {out_shapes};
}

std::vector<Type> InferDtypeForArgmin(const std::vector<Type> &inputs_type,
                                      const framework::AttrMapType &attrs) {
  CHECK(!inputs_type.empty())
      << "The input's type size is 0! Please check again.";
  return {Int(32)};
}

std::vector<std::vector<std::string>> InferLayoutForArgmin(
    const std::vector<framework::shape_t> &input_shapes,
    const std::vector<std::string> &input_layouts,
    const framework::NodeAttr &attrs,
    const Target &target) {
  CHECK_EQ(input_shapes.size(), 1U)
      << "The input's shape size is not 1! Please check again.";
  CHECK_EQ(input_layouts.size(), 1U)
      << "The input's layout size is not 1! Please check again.";
  return {input_layouts, input_layouts};
}
}  // namespace op
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(argmin_ops) {
  CINN_REGISTER_OP(argmin)
      .describe("This operator implements the op argmin.")
      .set_num_inputs(1)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>(
          "CINNStrategy", cinn::hlir::op::StrategyForArgmin)
      .set_attr("infershape",
                MakeOpFunction(cinn::hlir::op::InferShapeForArgmin))
      .set_attr("inferdtype",
                MakeOpFunction(cinn::hlir::op::InferDtypeForArgmin))
      .set_attr<cinn::hlir::framework::OpPatternKind>(
          "OpPattern", cinn::hlir::framework::OpPatternKind::kNonFusible)
      .set_support_level(4);

  return true;
}
