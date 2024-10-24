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

#include "absl/types/variant.h"
#include "glog/logging.h"
#include "paddle/cinn/common/cas.h"
#include "paddle/cinn/common/cinn_value.h"
#include "paddle/cinn/common/common.h"
#include "paddle/cinn/common/context.h"
#include "paddle/cinn/common/ir_util.h"
#include "paddle/cinn/common/macros.h"
#include "paddle/cinn/common/target.h"
#include "paddle/cinn/hlir/framework/op.h"
#include "paddle/cinn/hlir/framework/op_strategy.h"
#include "paddle/cinn/hlir/op/op_util.h"
#include "paddle/cinn/hlir/pe/elementwise.h"
#include "paddle/cinn/hlir/pe/ir_schedule_pe.h"
#include "paddle/cinn/hlir/pe/nn.h"
#include "paddle/cinn/hlir/pe/schedule.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/ir/tensor.h"
#include "paddle/cinn/lang/compute.h"
#include "paddle/cinn/lang/packed_func.h"
#include "paddle/cinn/poly/stage.h"
#include "paddle/common/enforce.h"

namespace cinn {
namespace hlir {
namespace op {

using cinn::common::CINNValue;
using cinn::common::CINNValuePack;

std::shared_ptr<framework::OpStrategy> StrategyForRandInt(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<int>> &output_shapes,
    const Target &target) {
  framework::CINNCompute randint_compute(
      [=](lang::Args args, lang::RetValue *ret) {
        PADDLE_ENFORCE_GE(
            attrs.attr_store.count("shape"),
            1,
            ::common::errors::NotFound("The attribute 'shape' is not found in "
                                       "attr_store. Please ensure it is set."));
        ir::Tensor shape_tensor;
        std::string tensor_name = "randint_out";
        auto out = pe::Identity(shape_tensor, tensor_name).front();
        std::vector<CINNValue> res{CINNValue(out)};
        *ret = CINNValuePack{res};
      });
  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(randint_compute,
                    GetInjectiveScheduleFunc(output_shapes, target),
                    "strategy.randint.x86",
                    1);
  return strategy;
}

}  // namespace op
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(randint_ops) {
  CINN_REGISTER_OP(randint)
      .describe("RandInt")
      .set_num_inputs(0)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>(
          "CINNStrategy", cinn::hlir::op::StrategyForRandInt)
      .set_attr<cinn::hlir::framework::OpPatternKind>(
          "OpPattern", cinn::hlir::framework::OpPatternKind::kNonFusible)
      .set_support_level(4);

  return true;
}
