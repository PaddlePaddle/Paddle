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

#include "paddle/cinn/hlir/op/contrib/argmax.h"

#include <iostream>
#include <vector>

#include "paddle/cinn/common/cinn_value.h"
#include "paddle/cinn/common/common.h"
#include "paddle/cinn/common/context.h"
#include "paddle/cinn/common/macros.h"
#include "paddle/cinn/hlir/framework/op.h"
#include "paddle/cinn/hlir/framework/op_strategy.h"
#include "paddle/cinn/hlir/op/contrib/sort.h"
#include "paddle/cinn/hlir/pe/ir_schedule_pe.h"
#include "paddle/cinn/hlir/pe/nn.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"
#include "paddle/cinn/ir/tensor.h"
#include "paddle/cinn/lang/compute.h"
#include "paddle/cinn/optim/ir_simplify.h"

namespace cinn {
namespace hlir {
namespace op {

using cinn::common::CINNValue;
using framework::shape_t;
using ir::Tensor;

std::vector<ir::Tensor> Argmax(const Tensor &in_tensor,
                               const cinn::common::Target &target,
                               const int &axis,
                               const bool &keep_dims,
                               const std::string &name) {
  auto shape = in_tensor->shape;
  auto ndim = shape.size();
  PADDLE_ENFORCE_GT(
      ndim,
      0,
      ::common::errors::InvalidArgument(
          "The dimension of input tensor must be greater than 0."));

  int pos_axis = axis;
  if (axis < 0) {
    pos_axis = static_cast<int>(ndim) + axis;
  }
  PADDLE_ENFORCE_LT(
      pos_axis,
      ndim,
      ::common::errors::InvalidArgument(
          "The axis must be less than the dimension of input tensor."));
  PADDLE_ENFORCE_GE(pos_axis,
                    0,
                    ::common::errors::InvalidArgument(
                        "The axis must be greater than or equal to 0."));

  std::vector<Expr> output_shape;
  for (int i = 0; i < shape.size(); ++i) {
    PADDLE_ENFORCE_EQ(shape[i].is_constant(),
                      true,
                      ::common::errors::InvalidArgument(
                          "Input tensor's shape should be constant value."));
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
      ArgSort(in_tensor, target, pos_axis, false, name + "_index");
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
  return {res, sort_index.at(0), sort_index.at(1)};
}

std::shared_ptr<framework::OpStrategy> StrategyForArgmax(
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
    PADDLE_THROW(::common::errors::Fatal("reduce dimension is not set!"));
  }
  if (attrs.attr_store.count("keep_dim")) {
    keep_dims = absl::get<bool>(attrs.attr_store.at("keep_dim"));
  }

  framework::CINNCompute argmax_compute([=](lang::Args args,
                                            lang::RetValue *ret) {
    PADDLE_ENFORCE_EQ(
        !args.empty(),
        true,
        ::common::errors::InvalidArgument(
            "The input argument of argmax compute is empty! Please check."));
    cinn::common::CINNValuePack pack_args = args[0];
    std::string tensor_name = UniqName("Argmax_out");
    PADDLE_ENFORCE_GE(pack_args.size(),
                      1U,
                      ::common::errors::InvalidArgument(
                          "There should be 1 input args for argmax compute"));
    Expr in_expr = pack_args[0];
    PADDLE_ENFORCE_NOT_NULL(
        in_expr.as_tensor(),
        ::common::errors::InvalidArgument(
            "The input argument of argmax compute is null."));
    Tensor in_tensor = in_expr.as_tensor_ref();
    PADDLE_ENFORCE_EQ(pack_args.size(),
                      2U,
                      ::common::errors::InvalidArgument(
                          "The input argument of argmax compute must be 2."));
    PADDLE_ENFORCE_EQ(
        pack_args[1].is_string(),
        true,
        ::common::errors::InvalidArgument(
            "The input argument of argmax compute must be string."));
    tensor_name = pack_args[1].operator std::string();
    std::vector<ir::Tensor> out_tensor =
        Argmax(in_tensor, target, axis, keep_dims, tensor_name);

    std::vector<CINNValue> cinn_values{CINNValue(out_tensor[0]),
                                       CINNValue(out_tensor[1]),
                                       CINNValue(out_tensor[2])};
    *ret = cinn::common::CINNValuePack{cinn_values};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(argmax_compute, "strategy.argmax", 1);

  return strategy;
}

}  // namespace op
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(argmax_ops) {
  CINN_REGISTER_OP(argmax)
      .describe("This operator implements the op argmax.")
      .set_num_inputs(1)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>(
          "CINNStrategy", cinn::hlir::op::StrategyForArgmax)
      .set_attr<cinn::hlir::framework::OpPatternKind>(
          "OpPattern", cinn::hlir::framework::OpPatternKind::kNonFusible)
      .set_support_level(4);

  return true;
}
