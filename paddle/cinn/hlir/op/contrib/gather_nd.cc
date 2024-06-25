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

#include "paddle/cinn/hlir/op/contrib/gather_nd.h"

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
#include "paddle/cinn/hlir/pe/elementwise.h"
#include "paddle/cinn/hlir/pe/ir_schedule_pe.h"
#include "paddle/cinn/hlir/pe/nn.h"
#include "paddle/cinn/hlir/pe/schedule.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/tensor.h"
#include "paddle/cinn/lang/builtin.h"
#include "paddle/cinn/lang/compute.h"

namespace cinn {
namespace hlir {
namespace op {

using cinn::common::CINNValue;
using cinn::common::CINNValuePack;

ir::Tensor GatherNd(const ir::Tensor &x,
                    const ir::Tensor &index,
                    const std::string &name) {
  std::vector<Expr> x_shape = x->shape;
  std::vector<Expr> index_shape = index->shape;
  size_t x_shape_size = x_shape.size();
  size_t index_shape_size = index_shape.size();
  std::vector<Expr> out_shape;
  out_shape.insert(out_shape.end(), index_shape.begin(), index_shape.end() - 1);
  out_shape.insert(out_shape.end(),
                   x_shape.begin() + index_shape.back().as_int32(),
                   x_shape.end());
  auto res = Compute(
      out_shape,
      [=](const std::vector<Expr> &indices) {
        std::vector<Expr> indices_position;
        for (size_t i = 0; i < index_shape_size - 1; ++i) {
          indices_position.push_back(
              ir::Cast::Make(cinn::common::Int(32), indices[i]));
        }
        indices_position.push_back(
            ir::Cast::Make(cinn::common::Int(32), Expr(0)));
        size_t indices_position_size = indices_position.size();
        std::vector<Expr> real_indices;
        for (size_t i = 0; i < index_shape.back().as_int32(); ++i) {
          indices_position[indices_position_size - 1] =
              ir::Cast::Make(cinn::common::Int(32), Expr(i));
          real_indices.push_back(
              ir::Cast::Make(cinn::common::Int(32), index(indices_position)));
        }
        if (real_indices.size() == x_shape_size) {
          return x(real_indices);
        }
        for (size_t i = index_shape_size - 1; i < indices.size(); ++i) {
          real_indices.push_back(indices[i]);
        }
        return x(real_indices);
      },
      name);
  return res;
}

ir::Tensor GatherNdSymbolic(const ir::Tensor &x,
                            const ir::Tensor &index,
                            const std::string &name) {
  std::vector<Expr> x_shape = x->shape;
  std::vector<Expr> index_shape = index->shape;
  size_t x_shape_size = x_shape.size();
  size_t index_shape_size = index_shape.size();
  std::vector<Expr> out_shape;
  out_shape.insert(out_shape.end(), index_shape.begin(), index_shape.end() - 1);
  out_shape.insert(out_shape.end(),
                   x_shape.begin() + index_shape.back().as_int64(),
                   x_shape.end());
  auto res = Compute(
      out_shape,
      [=](const std::vector<Expr> &indices) {
        std::vector<Expr> indices_position;
        for (size_t i = 0; i < index_shape_size - 1; ++i) {
          indices_position.push_back(
              ir::Cast::Make(cinn::common::Int(64), indices[i]));
        }
        indices_position.push_back(
            ir::Cast::Make(cinn::common::Int(64), Expr(0)));
        size_t indices_position_size = indices_position.size();
        std::vector<Expr> real_indices;
        for (size_t i = 0; i < index_shape.back().as_int64(); ++i) {
          indices_position[indices_position_size - 1] =
              ir::Cast::Make(cinn::common::Int(64), Expr(i));
          real_indices.push_back(
              ir::Cast::Make(cinn::common::Int(64), index(indices_position)));
        }
        if (real_indices.size() == x_shape_size) {
          return x(real_indices);
        }
        for (size_t i = index_shape_size - 1; i < indices.size(); ++i) {
          real_indices.push_back(indices[i]);
        }
        return x(real_indices);
      },
      name);
  return res;
}

std::shared_ptr<framework::OpStrategy> StrategyForGatherNd(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<int>> &output_shapes,
    const Target &target) {
  std::string op_name("gather_nd");

  framework::CINNCompute gather_nd_compute(
      [=](lang::Args args, lang::RetValue *ret) {
        CHECK(!args.empty()) << "The input arguments of " << op_name
                             << " compute is empty! Please check.\n";
        CINNValuePack pack_args = args[0];
        CHECK_GE(pack_args.size(), 2U)
            << "2 input tensors for " << op_name << " compute\n";
        Expr x = pack_args[0];
        Expr index = pack_args[1];
        CHECK(x.as_tensor());
        CHECK(index.as_tensor());
        CHECK(!output_shapes.empty());
        auto tensor_x = x.as_tensor_ref();
        auto tensor_index = index.as_tensor_ref();
        auto stages = CreateStages({tensor_x, tensor_index});
        VLOG(3) << "x shape: " << utils::Join(tensor_x->shape, ", ")
                << ", index shape: " << utils::Join(tensor_index->shape, ", ")
                << ", output_shapes: " << utils::Join(output_shapes[0], ", ");
        CHECK_EQ(pack_args.size(), 3U);
        std::string tensor_name = pack_args[2].operator std::string();
        ir::Tensor out = GatherNd(tensor_x, tensor_index, tensor_name);
        std::vector<CINNValue> res;
        stages->InsertLazily(out);
        res.push_back(CINNValue(out));
        CHECK(!out_type.empty())
            << "Output type of " << op_name << " is empty! Please check.\n";
        res.push_back(CINNValue(stages));
        *ret = CINNValuePack{res};
      });

  framework::CINNSchedule gather_nd_schedule([=](lang::Args args,
                                                 lang::RetValue *ret) {
    PADDLE_ENFORCE_EQ(
        !args.empty(),
        true,
        phi::errors::InvalidArgument("The input argument of gather_nd_schedule "
                                     "is empty! Please check.\n"));
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
        phi::errors::InvalidArgument(
            "The vec_ast of gather_nd_schedule is empty! Please check.\n"));
    ir::ModuleExpr mod_expr(vec_ast);
    ir::IRSchedule ir_sch(mod_expr);
    ir_sch.MergeExprs();
    int64_t prod_size = std::accumulate(output_shapes[0].begin(),
                                        output_shapes[0].end(),
                                        1,
                                        std::multiplies<int>());
    if (prod_size > 1) {
      target.arch.Match(
          [&](common::UnknownArch) { CINN_NOT_IMPLEMENTED; },
          [&](common::X86Arch) {
            pe::IRScheduleInjectiveCPU(
                ir_sch, output_shapes.front(), target, true);
          },
          [&](common::ARMArch) { CINN_NOT_IMPLEMENTED; },
          [&](common::NVGPUArch) {
            pe::IRGpuScheduleInjective(ir_sch, output_shapes.front(), target);
          },
          [&](common::HygonDCUArchHIP) {
            pe::IRGpuScheduleInjective(ir_sch, output_shapes.front(), target);
          });
    }
    std::vector<cinn::common::CINNValue> res{
        cinn::common::CINNValue(ir_sch.GetModule().GetExprs().at(0))};
    *ret = cinn::common::CINNValuePack{res};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(
      gather_nd_compute, gather_nd_schedule, "strategy.gather_nd.x86", 1);
  return strategy;
}

std::shared_ptr<framework::OpStrategy> StrategyForGatherNdSymbolic(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<ir::Dim>> &output_shapes,
    const Target &target) {
  std::string op_name("gather_nd");

  framework::CINNCompute gather_nd_compute(
      [=](lang::Args args, lang::RetValue *ret) {
        CHECK(!args.empty()) << "The input arguments of " << op_name
                             << " compute is empty! Please check.\n";
        CINNValuePack pack_args = args[0];
        CHECK_GE(pack_args.size(), 2U)
            << "2 input tensors for " << op_name << " compute\n";
        Expr x = pack_args[0];
        Expr index = pack_args[1];
        CHECK(x.as_tensor());
        CHECK(index.as_tensor());
        CHECK(!output_shapes.empty());
        auto tensor_x = x.as_tensor_ref();
        auto tensor_index = index.as_tensor_ref();
        auto stages = CreateStages({tensor_x, tensor_index});
        VLOG(3) << "x shape: " << utils::Join(tensor_x->shape, ", ")
                << ", index shape: " << utils::Join(tensor_index->shape, ", ")
                << ", output_shapes: " << utils::Join(output_shapes[0], ", ");
        CHECK_EQ(pack_args.size(), 3U);
        std::string tensor_name = pack_args[2].operator std::string();
        ir::Tensor out = GatherNdSymbolic(tensor_x, tensor_index, tensor_name);
        std::vector<CINNValue> res;
        stages->InsertLazily(out);
        res.push_back(CINNValue(out));
        CHECK(!out_type.empty())
            << "Output type of " << op_name << " is empty! Please check.\n";
        res.push_back(CINNValue(stages));
        *ret = CINNValuePack{res};
      });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(
      gather_nd_compute, lang::PackedFunc(), "strategy.gather_nd.x86", 1);
  return strategy;
}

std::vector<std::vector<int>> InferShapeForGatherNd(
    const std::vector<std::vector<int>> &inputs_shape,
    const framework::AttrMapType &attrs) {
  CHECK_EQ(inputs_shape.size(), 2U)
      << "The input's shape size should be 2! Please check again.";
  std::vector<int> x_shape = inputs_shape[0];
  std::vector<int> index_shape = inputs_shape[1];
  CHECK_GE(index_shape.size(), 1U) << "Index shape must greater or equal to 1!";
  CHECK_LE(index_shape.back(), x_shape.size())
      << "Index shape[-1] must be no more than x.rank! Please check again.";
  std::vector<int> output_shape;
  output_shape.insert(
      output_shape.end(), index_shape.begin(), index_shape.end() - 1);
  output_shape.insert(
      output_shape.end(), x_shape.begin() + index_shape.back(), x_shape.end());
  return {output_shape};
}

std::vector<Type> InferDtypeForGatherNd(const std::vector<Type> &inputs_type,
                                        const framework::AttrMapType &attrs) {
  CHECK(!inputs_type.empty())
      << "The input's type size is 0! Please check again.";
  std::vector<Type> res{inputs_type[0]};
  return res;
}

}  // namespace op
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(gather_nd_ops) {
  CINN_REGISTER_OP(gather_nd)
      .describe("GatherNd.")
      .set_num_inputs(2)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunctionSymbolic>(
          "CINNStrategySymbolic", cinn::hlir::op::StrategyForGatherNdSymbolic)
      .set_attr<cinn::hlir::framework::StrategyFunction>(
          "CINNStrategy", cinn::hlir::op::StrategyForGatherNd)
      .set_attr("infershape",
                MakeOpFunction(cinn::hlir::op::InferShapeForGatherNd))
      .set_attr("inferdtype",
                MakeOpFunction(cinn::hlir::op::InferDtypeForGatherNd))
      .set_attr<cinn::hlir::framework::OpPatternKind>(
          "OpPattern", cinn::hlir::framework::OpPatternKind::kInjective)
      .set_support_level(4);

  return true;
}
