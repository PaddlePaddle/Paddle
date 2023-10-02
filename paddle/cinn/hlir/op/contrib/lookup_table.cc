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
#include "paddle/cinn/hlir/framework/node.h"
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
#include "paddle/cinn/lang/builtin.h"
#include "paddle/cinn/lang/compute.h"
#include "paddle/utils/flags.h"

namespace cinn {
namespace hlir {
namespace op {

using common::CINNValue;
using common::CINNValuePack;

ir::Tensor LookupTable(const ir::Tensor& table,
                       const ir::Tensor& ids,
                       const int64_t padding_idx,
                       const std::string& output_name) {
  CHECK_EQ(table->shape.size(), 2);
  CHECK_GT(ids->shape.size(), 1);
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
        auto ids_offset = ir::Cast::Make(common::I32(), ids(offsets));
        auto pred = ir::And::Make(
            Expr(padding_idx != -1),
            ir::EQ::Make(ids_offset, Expr(static_cast<int32_t>(padding_idx))));
        return ir::Select::Make(pred,
                                ir::Cast::Make(table->type(), Expr(0)),
                                table(ids_offset, indices.back()));
      },
      common::UniqName(output_name));
}

std::shared_ptr<framework::OpStrategy> StrategyForLookupTable(
    const framework::NodeAttr& attrs,
    const std::vector<ir::Tensor>& inputs,
    const std::vector<Type>& out_type,
    const std::vector<std::vector<int>>& output_shapes,
    const Target& target) {
  std::string op_name("lookup_table");
  const auto& attr_store = attrs.attr_store;
  CHECK(attr_store.count("padding_idx")) << "find no attr of axis";
  auto padding_idx = absl::get<int64_t>(attr_store.at("padding_idx"));

  framework::CINNCompute lookup_table_compute([=](lang::Args args,
                                                  lang::RetValue* ret) {
    CHECK(!args.empty()) << "The input arguments of " << op_name
                         << " compute is empty! Please check.\n";
    CINNValuePack pack_args = args[0];
    CHECK_GE(pack_args.size(), 2U)
        << "2 input tensors for " << op_name << " compute\n";
    Expr A = pack_args[0];
    Expr B = pack_args[1];
    CHECK(A.as_tensor());
    CHECK(B.as_tensor());
    CHECK(!output_shapes.empty());
    auto tensor_A = A.as_tensor_ref();
    auto tensor_B = B.as_tensor_ref();
    auto stages = CreateStages({tensor_A, tensor_B});
    VLOG(3) << "A shape: " << utils::Join(tensor_A->shape, ", ")
            << ", B shape: " << utils::Join(tensor_B->shape, ", ")
            << ", output_shapes: " << utils::Join(output_shapes[0], ", ");
    CHECK_EQ(pack_args.size(), 3U);
    std::string tensor_name = pack_args[2].operator std::string();

    ir::Tensor out = LookupTable(tensor_A, tensor_B, padding_idx, tensor_name);
    std::vector<CINNValue> res;
    stages->InsertLazily(out);
    res.push_back(CINNValue(out));
    CHECK(!out_type.empty())
        << "Output type of " << op_name << " is empty! Please check.\n";
    res.push_back(CINNValue(stages));
    *ret = CINNValuePack{res};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(lookup_table_compute,
                    GetInjectiveScheduleFunc(output_shapes, target),
                    "strategy.lookup_table",
                    1);
  return strategy;
}

std::vector<framework::shape_t> InferShapeForLookupTable(
    const std::vector<framework::shape_t>& inputs_shape,
    const framework::AttrMapType& attrs) {
  CHECK(!inputs_shape.empty() && !inputs_shape[0].empty())
      << "The input's shape size is 0! Please check again.";

  auto res = inputs_shape[1];
  res.back() = inputs_shape[0].back();
  return {res};
}

std::vector<Type> InferDtypeForLookupTable(
    const std::vector<Type>& inputs_type, const framework::AttrMapType& attrs) {
  CHECK(!inputs_type.empty())
      << "The input's type size is 0! Please check again.";
  std::vector<Type> res{inputs_type[0]};
  return res;
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
      .set_attr("infershape",
                MakeOpFunction(cinn::hlir::op::InferShapeForLookupTable))
      .set_attr("inferdtype",
                MakeOpFunction(cinn::hlir::op::InferDtypeForLookupTable))
      .set_attr<cinn::hlir::framework::OpPatternKind>(
          "OpPattern", cinn::hlir::framework::OpPatternKind::kInjective);
  return true;
}
