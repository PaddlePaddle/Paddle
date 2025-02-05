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

#include "paddle/cinn/hlir/op/contrib/repeat.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "paddle/cinn/common/cas.h"
#include "paddle/cinn/common/common.h"
#include "paddle/cinn/common/context.h"
#include "paddle/cinn/common/macros.h"
#include "paddle/cinn/hlir/framework/op.h"
#include "paddle/cinn/hlir/framework/op_strategy.h"
#include "paddle/cinn/hlir/pe/ir_schedule_pe.h"
#include "paddle/cinn/hlir/pe/nn.h"
#include "paddle/cinn/hlir/pe/transform.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"
#include "paddle/cinn/ir/tensor.h"
#include "paddle/cinn/lang/compute.h"

namespace cinn {
namespace hlir {
namespace op {

using cinn::common::CINNValuePack;

std::vector<ir::Tensor> Repeat(const ir::Tensor &tensor,
                               int repeats,
                               int axis,
                               const std::string &output_name) {
  int ndim = static_cast<int>(tensor->shape.size());
  PADDLE_ENFORCE_EQ(-ndim - 1 <= axis && axis <= ndim,
                    true,
                    ::common::errors::InvalidArgument(
                        "The value of `axis` is out of the valid range. "
                        "Repeat only accepts `axis` in the range [-data.ndim - "
                        "1, data.ndim], "
                        "but got axis = %d, and data.ndim = %d. "
                        "Please check your input and ensure `axis` is within "
                        "the valid range.",
                        axis,
                        ndim));

  PADDLE_ENFORCE_GE(
      repeats,
      1,
      ::common::errors::InvalidArgument(
          "The value of `repeats` is less than 1. "
          "Repeat only accepts `repeats >= 1`, but got repeats = %d. "
          "Please check your input and ensure `repeats` is greater than or "
          "equal to 1.",
          repeats));
  if (axis < 0) {
    // Calculate offset from last dimension
    axis += ndim;
  }
  std::vector<Expr> new_shape;
  for (size_t i = 0; i < static_cast<size_t>(axis); ++i) {
    new_shape.push_back(tensor->shape[i]);
  }
  new_shape.push_back(repeats * tensor->shape[axis]);
  for (size_t i = axis + 1; i < tensor->shape.size(); ++i) {
    new_shape.push_back(tensor->shape[i]);
  }

  ir::Tensor res = lang::Compute(
      {new_shape},
      [=](const std::vector<ir::Expr> &indices) {
        std::vector<Expr> idx;
        for (size_t i = 0; i < static_cast<size_t>(axis); ++i) {
          idx.push_back(indices[i]);
        }
        idx.push_back(lang::FloorDivide(indices[axis], Expr(repeats)));
        for (size_t i = axis + 1; i < indices.size(); ++i) {
          idx.push_back(indices[i]);
        }
        return tensor(idx);
      },
      cinn::common::UniqName(output_name));
  return {res};
}

std::shared_ptr<framework::OpStrategy> StrategyForRepeat(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<int>> &output_shapes,
    const Target &target) {
  int repeats = 0;
  int axis = 0;
  for (auto &iter : attrs.attr_store) {
    if (iter.first == "repeats") {
      repeats = absl::get<int>(iter.second);
    } else if (iter.first == "axis") {
      axis = absl::get<int>(iter.second);
    }
  }
  PADDLE_ENFORCE_GE(
      repeats,
      1,
      ::common::errors::InvalidArgument(
          "The value of `repeats` is less than 1. "
          "Repeat only accepts `repeats >= 1`, but got repeats = %d. "
          "Please check your input and ensure `repeats` is greater than or "
          "equal to 1.",
          repeats));
  framework::CINNCompute repeat_compute([=](lang::Args args,
                                            lang::RetValue *ret) {
    PADDLE_ENFORCE_EQ(!args.empty(),
                      true,
                      ::common::errors::InvalidArgument(
                          "The input arguments of Repeat compute is empty. "
                          "Please check your input arguments and ensure they "
                          "are not empty."));

    CINNValuePack pack_args = args[0];

    PADDLE_ENFORCE_GE(
        pack_args.size(),
        1U,
        ::common::errors::InvalidArgument(
            "At least 1 input tensor is required for Repeat compute, "
            "but got %d input tensors. Please check your input.",
            pack_args.size()));

    Expr A = pack_args[0];
    PADDLE_ENFORCE_NOT_NULL(
        A.as_tensor(),
        ::common::errors::InvalidArgument(
            "The first argument in pack_args is null "
            "Please ensure the first argument is a valid tensor."));

    PADDLE_ENFORCE_EQ(
        !output_shapes.empty(),
        true,
        ::common::errors::InvalidArgument(
            "The output shapes are empty. "
            "Please ensure the output shapes are correctly specified."));

    auto tensor_A = A.as_tensor_ref();
    VLOG(3) << "A shape: " << utils::Join(tensor_A->shape, ", ")
            << ", output_shapes: " << utils::Join(output_shapes[0], ", ");

    PADDLE_ENFORCE_EQ(
        pack_args.size(),
        2U,
        ::common::errors::InvalidArgument(
            "[Error info] The size of pack_args should equal to 2."));
    std::string tensor_name = pack_args[1].operator std::string();

    std::vector<ir::Tensor> out = Repeat(tensor_A, repeats, axis, tensor_name);
    PADDLE_ENFORCE_EQ(
        out.size(),
        1U,
        ::common::errors::InvalidArgument(
            "The size of Repeat's output should be 1, but got %d. "
            "Please check your Repeat function implementation.",
            out.size()));

    std::vector<cinn::common::CINNValue> res;
    for (auto &t : out) {
      res.push_back(cinn::common::CINNValue(t));
    }

    *ret = cinn::common::CINNValuePack{res};
  });

  framework::CINNSchedule repeat_schedule([=](lang::Args args,
                                              lang::RetValue *ret) {
    PADDLE_ENFORCE_EQ(
        !args.empty(),
        true,
        ::common::errors::InvalidArgument(
            "The input argument of repeat schedule is empty. "
            "Please check your input arguments and ensure they are "
            "not empty."));
    cinn::common::CINNValuePack arg_pack = args[0];
    std::vector<Expr> vec_ast;
    for (int i = 0; i < arg_pack.size(); i++) {
      if (arg_pack[i].is_expr()) {
        Expr temp = arg_pack[i];
        vec_ast.emplace_back(temp);
      }
    }
    PADDLE_ENFORCE_EQ(
        !vec_ast.empty(),
        true,
        ::common::errors::InvalidArgument(
            "The vector of AST expressions is empty. "
            "Please ensure there are valid expressions in the argument pack."));
    ir::ModuleExpr mod_expr(vec_ast);
    ir::IRSchedule ir_sch(mod_expr);
    ir_sch.MergeExprs();
    int64_t prod_size = std::accumulate(output_shapes[0].begin(),
                                        output_shapes[0].end(),
                                        1,
                                        std::multiplies<int>());
    if (prod_size > 1) {
      target.arch.Match(
          [&](std::variant<common::UnknownArch, common::ARMArch>) {
            CINN_NOT_IMPLEMENTED;
          },
          [&](common::X86Arch) {
            pe::IRScheduleInjectiveCPU(
                ir_sch, output_shapes.front(), target, true);
          },
          [&](common::NVGPUArch) {
            pe::IRGpuScheduleInjective(ir_sch, output_shapes.front(), target);
          },
          [&](std::variant<common::HygonDCUArchHIP, common::HygonDCUArchSYCL>) {
            pe::IRGpuScheduleInjective(ir_sch, output_shapes.front(), target);
          });
    }
    std::vector<cinn::common::CINNValue> res{
        cinn::common::CINNValue(ir_sch.GetModule().GetExprs().at(0))};
    *ret = cinn::common::CINNValuePack{res};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(repeat_compute, repeat_schedule, "strategy.repeat.x86", 1);

  return strategy;
}

}  // namespace op
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(repeat_ops) {
  CINN_REGISTER_OP(repeat)
      .describe("Repeat elements of an array `repeats` times along axis `axis`")
      .set_num_inputs(1)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>(
          "CINNStrategy", cinn::hlir::op::StrategyForRepeat)
      .set_support_level(4);

  return true;
}
