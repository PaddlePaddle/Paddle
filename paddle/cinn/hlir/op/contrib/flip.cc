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

#include "paddle/cinn/hlir/op/contrib/flip.h"

#include <gflags/gflags.h>

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
#include "paddle/cinn/hlir/op/op_util.h"
#include "paddle/cinn/hlir/pe/elementwise.h"
#include "paddle/cinn/hlir/pe/ir_schedule_pe.h"
#include "paddle/cinn/hlir/pe/transform.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/ir_schedule.h"
#include "paddle/cinn/ir/tensor.h"
#include "paddle/cinn/lang/builtin.h"
#include "paddle/cinn/lang/compute.h"

DECLARE_bool(cinn_ir_schedule);

namespace cinn {
namespace hlir {
namespace op {

using common::CINNValue;
using common::CINNValuePack;
using framework::shape_t;

ir::Tensor Flip(const ir::Tensor &input,
                const std::vector<int> &axes,
                const std::string &name) {
  return cinn::hlir::pe::Reverse(input, axes, name);
}

std::shared_ptr<framework::OpStrategy> StrategyForFlip(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<int>> &output_shapes,
    const Target &target) {
  CHECK(attrs.attr_store.count("axes")) << "find no attr of axes";
  std::vector<int> axes =
      absl::get<std::vector<int>>(attrs.attr_store.at("axes"));
  std::string op_name("flip");

  framework::CINNCompute flip_compute(
      [=](lang::Args args, lang::RetValue *ret) {
        CHECK(!args.empty()) << "The input argument of " << op_name
                             << " compute is empty! Please check.";
        CINNValuePack pack_args = args[0];
        CHECK_GE(pack_args.size(), 1U)
            << "1 input tensor for " << op_name << " compute";
        std::string tensor_name = UniqName(op_name + "_Out");
        if (FLAGS_cinn_ir_schedule) {
          CHECK_EQ(pack_args.size(), 2U);
          tensor_name = pack_args[1].operator std::string();
        }
        Expr A_expr = pack_args[0];
        CHECK(A_expr.as_tensor());
        ir::Tensor A = A_expr.as_tensor_ref();
        auto out = Flip(A, axes, tensor_name);
        auto stages = CreateStages({A});
        std::vector<CINNValue> res;
        stages->InsertLazily(out);
        res.push_back(CINNValue(out));
        res.push_back(CINNValue(stages));
        *ret = CINNValuePack{res};
      });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(flip_compute,
                    GetInjectiveScheduleFunc(output_shapes, target),
                    "strategy.flip.x86",
                    1);
  return strategy;
}

std::vector<shape_t> InferShapeForFlip(const std::vector<shape_t> &inputs_shape,
                                       const framework::AttrMapType &attrs) {
  CHECK_EQ(inputs_shape.size(), 1U)
      << "The input's shape size should be 1! Please check again.";
  std::vector<shape_t> res{inputs_shape[0]};
  return res;
}

std::vector<Type> InferDtypeForFlip(const std::vector<Type> &inputs_type,
                                    const framework::AttrMapType &attrs) {
  CHECK(!inputs_type.empty())
      << "The input's type size is 0! Please check again.";
  std::vector<Type> res{inputs_type[0]};
  return res;
}

}  // namespace op
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(flip_ops) {
  CINN_REGISTER_OP(flip)
      .describe("Flip.")
      .set_num_inputs(1)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>(
          "CINNStrategy", cinn::hlir::op::StrategyForFlip)
      .set_attr("infershape", MakeOpFunction(cinn::hlir::op::InferShapeForFlip))
      .set_attr("inferdtype", MakeOpFunction(cinn::hlir::op::InferDtypeForFlip))
      .set_support_level(4);

  return true;
}
