// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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
#include "paddle/cinn/hlir/framework/op.h"
#include "paddle/cinn/hlir/framework/op_strategy.h"
#include "paddle/cinn/hlir/op/op_util.h"
#include "paddle/cinn/hlir/pe/elementwise.h"
#include "paddle/cinn/hlir/pe/ir_schedule_pe.h"
#include "paddle/cinn/hlir/pe/transform.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"
#include "paddle/cinn/ir/tensor.h"
#include "paddle/cinn/lang/compute.h"

namespace cinn {
namespace hlir {
namespace op {

using cinn::common::CINNValue;
using cinn::common::CINNValuePack;
using framework::shape_t;

ir::Tensor BitcastConvert(const ir::Tensor &input,
                          const Type &dtype,
                          const std::string &name) {
  auto res = Compute(
      input->shape,
      [=](const std::vector<Expr> &indices) { return input(indices); },
      name);
  return res;
}

std::shared_ptr<framework::OpStrategy> StrategyForBitcastConvert(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<int>> &output_shapes,
    const Target &target) {
  std::string op_name("bitcast_convert");

  framework::CINNCompute bitcast_convert_compute([=](lang::Args args,
                                                     lang::RetValue *ret) {
    PADDLE_ENFORCE_EQ(
        !args.empty(),
        true,
        ::common::errors::InvalidArgument(
            "The input argument of %s compute is empty!", op_name));
    CINNValuePack pack_args = args[0];
    PADDLE_ENFORCE_GE(pack_args.size(),
                      1U,
                      ::common::errors::InvalidArgument(
                          "The size of pack_args should be greater than 0 . "));
    std::string tensor_name = UniqName(op_name + "_Out");
    Expr A_expr = pack_args[0];
    PADDLE_ENFORCE_NOT_NULL(A_expr.as_tensor(),
                            ::common::errors::InvalidArgument(
                                "The input argument A  is not a tensor."));
    ir::Tensor A = A_expr.as_tensor_ref();
    auto out = BitcastConvert(A, out_type[0], tensor_name);
    std::vector<CINNValue> res;
    res.push_back(CINNValue(out));
    *ret = CINNValuePack{res};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(bitcast_convert_compute,
                    GetInjectiveScheduleFunc(output_shapes, target),
                    "strategy.bitcast_convert.x86",
                    1);
  return strategy;
}

}  // namespace op
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(bitcast_convert_ops) {
  CINN_REGISTER_OP(bitcast_convert)
      .describe("BitcastConvert")
      .set_num_inputs(1)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>(
          "CINNStrategy", cinn::hlir::op::StrategyForBitcastConvert)
      .set_attr<cinn::hlir::framework::OpPatternKind>(
          "OpPattern", cinn::hlir::framework::OpPatternKind::kInjective)
      .set_support_level(4);

  return true;
}
