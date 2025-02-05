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

#include "paddle/cinn/common/cas.h"
#include "paddle/cinn/common/common.h"
#include "paddle/cinn/common/context.h"
#include "paddle/cinn/common/macros.h"
#include "paddle/cinn/common/target.h"
#include "paddle/cinn/hlir/framework/op.h"
#include "paddle/cinn/hlir/framework/op_strategy.h"
#include "paddle/cinn/hlir/op/contrib/logical_right_shift.h"
#include "paddle/cinn/hlir/op/op_util.h"
#include "paddle/cinn/hlir/pe/ir_schedule_pe.h"
#include "paddle/cinn/hlir/pe/nn.h"
#include "paddle/cinn/hlir/pe/schedule.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/ir/tensor.h"
#include "paddle/cinn/lang/compute.h"
#include "paddle/common/flags.h"

namespace cinn {
namespace hlir {
namespace op {

using cinn::common::_CINNValuePack_;
using cinn::common::CINNValue;
using cinn::common::CINNValuePack;
using framework::OpStrategy;
using framework::shape_t;
using framework::StrategyFunction;

ir::Tensor LogicalRightShift(const ir::Tensor &A,
                             const ir::Tensor &B,
                             const Target &target,
                             const std::string &output_name) {
  std::string extern_func = "cinn_";
  target.arch.Match(
      [&](common::X86Arch) { extern_func += "host_"; },
      [&](common::NVGPUArch) { extern_func += "nvgpu_"; },
      [&](std::variant<common::UnknownArch, common::X86Arch, common::ARMArch>) {
        CINN_NOT_IMPLEMENTED
      },
      [&](common::HygonDCUArchHIP) { extern_func += "hip_"; },
      [&](common::HygonDCUArchSYCL) { extern_func += "sycl_"; });

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

  framework::CINNCompute logical_right_shift_compute([=](lang::Args args,
                                                         lang::RetValue *ret) {
    PADDLE_ENFORCE_NE(
        args.empty(),
        true,
        ::common::errors::InvalidArgument(
            "The input argument of  %s compute is empty! Please check.\n",
            op_name.c_str()));
    CINNValuePack pack_args = args[0];
    PADDLE_ENFORCE_GE(pack_args.size(),
                      2U,
                      ::common::errors::InvalidArgument(
                          "2 input tensors for %s compute\n", op_name.c_str()));

    Expr A_expr = pack_args[0];
    Expr B_expr = pack_args[1];
    PADDLE_ENFORCE_EQ(
        A_expr.as_tensor(),
        true,
        ::common::errors::InvalidArgument("Input is not a Tensor"));
    PADDLE_ENFORCE_EQ(
        B_expr.as_tensor(),
        true,
        ::common::errors::InvalidArgument("Input in not a Tensor"));
    ir::Tensor A = A_expr.as_tensor_ref();
    ir::Tensor B = B_expr.as_tensor_ref();

    PADDLE_ENFORCE_EQ(pack_args.size(),
                      3U,
                      ::common::errors::InvalidArgument(
                          "Input size is %u not 3", pack_args.size()));
    std::string tensor_name = pack_args[2].operator std::string();

    auto out = LogicalRightShift(A, B, target, tensor_name);
    *ret = CINNValuePack{{CINNValue(Expr(out.get()))}};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(logical_right_shift_compute,
                    GetInjectiveScheduleFunc(output_shapes, target),
                    "strategy.logical_right_shift.x86",
                    1);
  return strategy;
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
      .set_attr<cinn::hlir::framework::OpPatternKind>(
          "OpPattern", cinn::hlir::framework::OpPatternKind::kElementWise)
      .set_support_level(4);

  return true;
}
