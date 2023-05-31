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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "cinn/common/cas.h"
#include "cinn/common/common.h"
#include "cinn/common/context.h"
#include "cinn/common/macros.h"
#include "cinn/common/target.h"
#include "cinn/hlir/framework/node.h"
#include "cinn/hlir/framework/op.h"
#include "cinn/hlir/framework/op_strategy.h"
#include "cinn/hlir/op/op_util.h"
#include "cinn/hlir/pe/ir_schedule_pe.h"
#include "cinn/hlir/pe/nn.h"
#include "cinn/hlir/pe/schedule.h"
#include "cinn/ir/ir.h"
#include "cinn/ir/ir_base.h"
#include "cinn/ir/ir_operators.h"
#include "cinn/ir/tensor.h"
#include "cinn/lang/builtin.h"
#include "cinn/lang/compute.h"
#include "gflags/gflags.h"

DECLARE_bool(cinn_ir_schedule);

namespace cinn {
namespace hlir {
namespace op {

using common::_CINNValuePack_;
using common::CINNValue;
using common::CINNValuePack;
using framework::OpStrategy;
using framework::shape_t;
using framework::StrategyFunction;

ir::Tensor LogicalRightShift(const ir::Tensor &A,
                             const ir::Tensor &B,
                             const Target &target,
                             const std::string &output_name) {
  std::string extern_func = "cinn_";
  if (target == common::DefaultHostTarget()) {
    extern_func += "host_";
  } else if (target == common::DefaultNVGPUTarget()) {
    extern_func += "nvgpu_";
  } else {
    CINN_NOT_IMPLEMENTED
  }

  extern_func += "logical_right_shift";

  if (A->type().is_int(32) || A->type().is_uint(32)) {
    extern_func += "_int32";
  } else {
    CINN_NOT_IMPLEMENTED
  }

  return Compute(
      A->shape,
      [=](const std::vector<Expr> &indices) {
        Expr x = A(indices);
        Expr y = B(indices);
        return lang::CallExtern(extern_func, {x, y});
      },
      output_name);
}

std::shared_ptr<OpStrategy> StrategyForLogicalRightShift(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<int>> &output_shapes,
    const Target &target) {
  std::string op_name("logical_right_shift");

  framework::CINNCompute logical_right_shift_compute(
      [=](lang::Args args, lang::RetValue *ret) {
        CHECK(!args.empty()) << "The input argument of " << op_name
                             << " compute is empty! Please check.\n";
        CINNValuePack pack_args = args[0];
        CHECK_GE(pack_args.size(), 2U)
            << "2 input tensors for " << op_name << " compute\n";

        Expr A_expr = pack_args[0];
        Expr B_expr = pack_args[1];
        CHECK(A_expr.as_tensor());
        CHECK(B_expr.as_tensor());
        ir::Tensor A = A_expr.as_tensor_ref();
        ir::Tensor B = B_expr.as_tensor_ref();

        std::string tensor_name = UniqName("T_LogicalRightShift_out");

        if (FLAGS_cinn_ir_schedule) {
          CHECK_EQ(pack_args.size(), 3U);
          tensor_name = pack_args[2].operator std::string();
        }

        auto out = LogicalRightShift(A, B, target, tensor_name);
        auto stages = CreateStages({out});
        *ret = CINNValuePack{{CINNValue(Expr(out.get())), CINNValue(stages)}};
      });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(logical_right_shift_compute,
                    GetInjectiveScheduleFunc(output_shapes, target),
                    "strategy.logical_right_shift.x86",
                    1);
  return strategy;
}

std::vector<framework::shape_t> InferShapeForLogicalRightShift(
    const std::vector<framework::shape_t> &inputs_shape,
    const framework::AttrMapType &attrs) {
  CHECK_EQ(inputs_shape.size(), 2U)
      << "The input's shape size should be 2! Please check again.";
  CHECK_EQ(inputs_shape[0].size(), inputs_shape[1].size())
      << "The inputs' dims should be equal.";
  std::vector<framework::shape_t> res{inputs_shape[0]};
  return res;
}

std::vector<Type> InferDtypeForLogicalRightShift(
    const std::vector<Type> &inputs_type, const framework::AttrMapType &attrs) {
  CHECK_EQ(inputs_type.size(), 2UL)
      << "The logical_right_shift op should has two inputs! Please check.";
  CHECK_EQ(inputs_type[0], inputs_type[1])
      << "The data type of input tensors of logical_right_shift op should be "
         "equal, but here x:"
      << inputs_type[0] << " != y:" << inputs_type[1] << "! Please check.";
  std::vector<Type> res{inputs_type[0]};
  return res;
}

}  // namespace op
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(logical_right_shift_ops) {
  CINN_REGISTER_OP(logical_right_shift)
      .describe("Logical Right Shift.")
      .set_num_inputs(2)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>(
          "CINNStrategy", cinn::hlir::op::StrategyForLogicalRightShift)
      .set_attr("infershape",
                MakeOpFunction(cinn::hlir::op::InferShapeForLogicalRightShift))
      .set_attr("inferdtype",
                MakeOpFunction(cinn::hlir::op::InferDtypeForLogicalRightShift))
      .set_attr<cinn::hlir::framework::OpPatternKind>(
          "OpPattern", cinn::hlir::framework::OpPatternKind::kElementWise)
      .set_support_level(4);

  return true;
}
