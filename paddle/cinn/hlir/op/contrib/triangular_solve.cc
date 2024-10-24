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
#include <vector>

#include "paddle/cinn/common/common.h"
#include "paddle/cinn/common/macros.h"
#include "paddle/cinn/hlir/framework/op.h"
#include "paddle/cinn/hlir/framework/op_strategy.h"
#include "paddle/cinn/hlir/op/op_util.h"
#include "paddle/cinn/hlir/pe/elementwise.h"
#include "paddle/cinn/hlir/pe/ir_schedule_pe.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/tensor.h"
#include "paddle/cinn/lang/compute.h"
#include "paddle/common/enforce.h"

namespace cinn {
namespace hlir {
namespace op {

using cinn::common::CINNValue;
using cinn::common::CINNValuePack;

std::shared_ptr<framework::OpStrategy> StrategyForTriangularSolve(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<int>> &output_shapes,
    const Target &target) {
  framework::CINNCompute triangular_solve_compute(
      [=](lang::Args args, lang::RetValue *ret) {
        PADDLE_ENFORCE_EQ(!args.empty(),
                          true,
                          ::common::errors::InvalidArgument(
                              "The input argument of triangular_solve is "
                              "empty. Please check the arguments."));
        CINNValuePack pack_args = args[0];
        PADDLE_ENFORCE_GE(
            pack_args.size(),
            2U,
            ::common::errors::InvalidArgument(
                "Two input tensors are required for the computation of "
                "triangular_solve, but received %d.",
                pack_args.size()));
        Expr a_expr = pack_args[0];
        Expr b_expr = pack_args[1];
        ir::Tensor a = a_expr.as_tensor_ref();
        ir::Tensor b = b_expr.as_tensor_ref();
        std::string tensor_name = "triangular_solve_out";
        auto out = pe::Identity(b, tensor_name).front();
        std::vector<CINNValue> res{CINNValue(out)};
        *ret = CINNValuePack{res};
      });
  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(triangular_solve_compute,
                    GetInjectiveScheduleFunc(output_shapes, target),
                    "strategy.triangular_solve.x86",
                    1);
  return strategy;
}

}  // namespace op
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(triangular_solve_ops) {
  CINN_REGISTER_OP(triangular_solve)
      .describe("TriangularSolve")
      .set_num_inputs(2)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>(
          "CINNStrategy", cinn::hlir::op::StrategyForTriangularSolve)
      .set_attr<cinn::hlir::framework::OpPatternKind>(
          "OpPattern", cinn::hlir::framework::OpPatternKind::kNonFusible)
      .set_support_level(4);

  return true;
}
