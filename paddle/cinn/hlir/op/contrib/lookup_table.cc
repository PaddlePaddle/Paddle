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

#include "paddle/cinn/hlir/op/contrib/lookup_table.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "paddle/cinn/common/cas.h"
#include "paddle/cinn/common/common.h"
#include "paddle/cinn/common/context.h"
#include "paddle/cinn/common/macros.h"
#include "paddle/cinn/common/type.h"
#include "paddle/cinn/hlir/framework/op.h"
#include "paddle/cinn/hlir/framework/op_strategy.h"
#include "paddle/cinn/hlir/op/op_util.h"
#include "paddle/cinn/hlir/pe/elementwise.h"
#include "paddle/cinn/hlir/pe/ir_schedule_pe.h"
#include "paddle/cinn/hlir/pe/nn.h"
#include "paddle/cinn/hlir/pe/schedule.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/tensor.h"
#include "paddle/cinn/lang/compute.h"
#include "paddle/common/flags.h"

namespace cinn {
namespace hlir {
namespace op {

using cinn::common::CINNValue;
using cinn::common::CINNValuePack;

ir::Tensor LookupTable(const ir::Tensor& table,
                       const ir::Tensor& ids,
                       const int64_t padding_idx,
                       const std::string& output_name) {
  PADDLE_ENFORCE_EQ(
      table->shape.size(),
      2,
      ::common::errors::InvalidArgument("The shape of table should be 2."));
  PADDLE_ENFORCE_GT(ids->shape.size(),
                    1,
                    ::common::errors::InvalidArgument(
                        "The shape of ids should be greater than 1."));
  auto output_shape = ids->shape;
  output_shape.back() = table->shape.back();

  return lang::Compute(
      output_shape,
      [&](const std::vector<ir::Expr>& indices) {
        std::vector<Expr> offsets;
        for (int i = 0; i < indices.size() - 1; ++i) {
          offsets.emplace_back(indices[i]);
        }
        offsets.emplace_back(Expr(0));
        // Because the current conversion rules have not been completed, static
        // conversion is done here.
        auto ids_offset = ir::Cast::Make(cinn::common::I32(), ids(offsets));
        auto pred = ir::And::Make(
            Expr(padding_idx != -1),
            ir::EQ::Make(ids_offset, Expr(static_cast<int32_t>(padding_idx))));
        return ir::Select::Make(pred,
                                ir::Cast::Make(table->type(), Expr(0)),
                                table(ids_offset, indices.back()));
      },
      cinn::common::UniqName(output_name));
}

std::shared_ptr<framework::OpStrategy> StrategyForLookupTable(
    const framework::NodeAttr& attrs,
    const std::vector<ir::Tensor>& inputs,
    const std::vector<Type>& out_type,
    const std::vector<std::vector<int>>& output_shapes,
    const Target& target) {
  std::string op_name("lookup_table");
  const auto& attr_store = attrs.attr_store;
  PADDLE_ENFORCE_EQ(attr_store.count("padding_idx"),
                    true,
                    ::common::errors::InvalidArgument(
                        "The padding_idx should be set in lookup_table."));
  auto padding_idx = absl::get<int64_t>(attr_store.at("padding_idx"));

  framework::CINNCompute lookup_table_compute([=](lang::Args args,
                                                  lang::RetValue* ret) {
    PADDLE_ENFORCE_EQ(
        !args.empty(),
        true,
        ::common::errors::InvalidArgument("The input arguments of lookup_table "
                                          "compute is empty! Please check."));
    CINNValuePack pack_args = args[0];
    PADDLE_ENFORCE_GE(pack_args.size(),
                      2U,
                      ::common::errors::InvalidArgument(
                          "The input arguments' size should be greater "
                          "than 2"));
    Expr A = pack_args[0];
    Expr B = pack_args[1];
    PADDLE_ENFORCE_NOT_NULL(A.as_tensor(),
                            ::common::errors::InvalidArgument(
                                "The input argument of lookup_table compute "
                                "is not tensor."));
    PADDLE_ENFORCE_NOT_NULL(B.as_tensor(),
                            ::common::errors::InvalidArgument(
                                "The input argument of lookup_table compute "
                                "is not tensor."));
    PADDLE_ENFORCE_EQ(!output_shapes.empty(),
                      true,
                      ::common::errors::InvalidArgument(
                          "The output_shapes should not be empty."));
    auto tensor_A = A.as_tensor_ref();
    auto tensor_B = B.as_tensor_ref();
    VLOG(3) << "A shape: " << utils::Join(tensor_A->shape, ", ")
            << ", B shape: " << utils::Join(tensor_B->shape, ", ")
            << ", output_shapes: " << utils::Join(output_shapes[0], ", ");
    PADDLE_ENFORCE_EQ(pack_args.size(),
                      3U,
                      ::common::errors::InvalidArgument(
                          "The input arguments' size should be 3"));
    std::string tensor_name = pack_args[2].operator std::string();

    ir::Tensor out = LookupTable(tensor_A, tensor_B, padding_idx, tensor_name);
    std::vector<CINNValue> res;
    res.push_back(CINNValue(out));
    PADDLE_ENFORCE_EQ(!out_type.empty(),
                      true,
                      ::common::errors::InvalidArgument(
                          "The output type of lookup_table is empty."));
    *ret = CINNValuePack{res};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(lookup_table_compute,
                    GetInjectiveScheduleFunc(output_shapes, target),
                    "strategy.lookup_table",
                    1);
  return strategy;
}

}  // namespace op
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(lookup_table_ops) {
  CINN_REGISTER_OP(lookup_table)
      .describe("Lookup table Operator.")
      .set_num_inputs(1)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>(
          "CINNStrategy", cinn::hlir::op::StrategyForLookupTable)
      .set_attr<cinn::hlir::framework::OpPatternKind>(
          "OpPattern", cinn::hlir::framework::OpPatternKind::kInjective);
  return true;
}
