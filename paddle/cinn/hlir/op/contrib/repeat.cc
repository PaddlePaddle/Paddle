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
#include "paddle/cinn/hlir/framework/node.h"
#include "paddle/cinn/hlir/framework/op.h"
#include "paddle/cinn/hlir/framework/op_strategy.h"
#include "paddle/cinn/hlir/pe/ir_schedule_pe.h"
#include "paddle/cinn/hlir/pe/nn.h"
#include "paddle/cinn/hlir/pe/transform.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"
#include "paddle/cinn/ir/tensor.h"
#include "paddle/cinn/lang/builtin.h"
#include "paddle/cinn/lang/compute.h"

namespace cinn {
namespace hlir {
namespace op {

using common::CINNValuePack;

std::vector<ir::Tensor> Repeat(const ir::Tensor &tensor,
                               int repeats,
                               int axis,
                               const std::string &output_name) {
  int ndim = static_cast<int>(tensor->shape.size());
  CHECK(-ndim - 1 <= axis && axis <= ndim)
      << "repeat only accepts `axis` in [-data.ndim - 1, data.ndim]"
      << ", but got axis = " << axis << ", and data.ndim = " << ndim;
  CHECK(repeats >= 1) << "repeat only accepts `repeats >= 1`"
                      << ", but got repeats = " << repeats;

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
      common::UniqName(output_name));
  return {res};
}

std::vector<std::vector<int>> InferShapeForRepeat(
    const std::vector<std::vector<int>> &inputs_shape,
    const framework::AttrMapType &attrs) {
  CHECK_EQ(inputs_shape.size(), 1U)
      << "The input's shape size should be 1! Please check again.";

  int repeats = 0;
  int axis = 0;
  std::vector<int> new_shape;
  const std::vector<int> &tensor_shape = inputs_shape[0];
  int ndim = static_cast<int>(tensor_shape.size());

  if (attrs.find("repeats") != attrs.end()) {
    repeats = absl::get<int>(attrs.at("repeats"));
  }
  if (attrs.find("axis") != attrs.end()) {
    axis = absl::get<int>(attrs.at("axis"));
  }

  if (axis < 0) {
    // Calculate offset from last dimension
    axis += ndim;
  }

  for (size_t i = 0; i < static_cast<size_t>(axis); ++i) {
    new_shape.push_back(tensor_shape[i]);
  }
  new_shape.push_back(repeats * tensor_shape[axis]);
  for (size_t i = axis + 1; i < tensor_shape.size(); ++i) {
    new_shape.push_back(tensor_shape[i]);
  }

  std::vector<std::vector<int>> res{new_shape};
  return res;
}

std::vector<Type> InferDtypeForRepeat(const std::vector<Type> &inputs_type,
                                      const framework::AttrMapType &attrs) {
  CHECK(!inputs_type.empty())
      << "The input's type size is 0! Please check again.";
  std::vector<Type> res{inputs_type[0]};
  return res;
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

  CHECK(repeats >= 1) << "repeat only accepts `repeats >= 1`"
                      << ", but got repeats = " << repeats;

  framework::CINNCompute repeat_compute([=](lang::Args args,
                                            lang::RetValue *ret) {
    CHECK(!args.empty())
        << "The input arguments of Repeat compute is empty! Please check.\n";
    CINNValuePack pack_args = args[0];
    CHECK_GE(pack_args.size(), 1U)
        << "at least 1 input tensors for Repeat compute\n";
    Expr A = pack_args[0];
    CHECK(A.as_tensor());
    CHECK(!output_shapes.empty());
    auto tensor_A = A.as_tensor_ref();
    VLOG(3) << "A shape: " << utils::Join(tensor_A->shape, ", ")
            << ", output_shapes: " << utils::Join(output_shapes[0], ", ");

    CHECK_EQ(pack_args.size(), 2U);
    std::string tensor_name = pack_args[1].operator std::string();

    std::vector<ir::Tensor> out = Repeat(tensor_A, repeats, axis, tensor_name);
    CHECK(out.size() == 1U) << "The size of Repeat's output should be 1";

    std::vector<common::CINNValue> res;
    auto stages = CreateStages({tensor_A});
    for (auto &t : out) {
      stages->InsertLazily(t);
      res.push_back(common::CINNValue(t));
    }

    res.push_back(common::CINNValue(stages));
    *ret = common::CINNValuePack{res};
  });

  framework::CINNSchedule repeat_schedule([=](lang::Args args,
                                              lang::RetValue *ret) {
    CHECK(!args.empty())
        << "The input argument of repeat schedule is empty! Please check.\n";
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
    int64_t prod_size = std::accumulate(output_shapes[0].begin(),
                                        output_shapes[0].end(),
                                        1,
                                        std::multiplies<int>());
    if (prod_size > 1) {
      if (target.arch == Target::Arch::NVGPU) {
        pe::IRCudaScheduleInjective(ir_sch, output_shapes.front(), target);
      } else if (target.arch == Target::Arch::X86) {
        pe::IRScheduleInjectiveCPU(ir_sch, output_shapes.front(), target, true);
      }
    }
    std::vector<common::CINNValue> res{
        common::CINNValue(ir_sch.GetModule().GetExprs().at(0))};
    *ret = common::CINNValuePack{res};
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
      .set_attr("infershape",
                MakeOpFunction(cinn::hlir::op::InferShapeForRepeat))
      .set_attr("inferdtype",
                MakeOpFunction(cinn::hlir::op::InferDtypeForRepeat))
      .set_support_level(4);

  return true;
}
