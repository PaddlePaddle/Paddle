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
#include "paddle/cinn/hlir/framework/op.h"
#include "paddle/cinn/hlir/framework/op_strategy.h"
#include "paddle/cinn/hlir/pe/elementwise.h"
#include "paddle/cinn/hlir/pe/ir_schedule_pe.h"
#include "paddle/cinn/hlir/pe/nn.h"
#include "paddle/cinn/hlir/pe/schedule.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/tensor.h"
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

  framework::CINNCompute gather_nd_compute([=](lang::Args args,
                                               lang::RetValue *ret) {
    PADDLE_ENFORCE_NE(
        args.empty(),
        true,
        ::common::errors::InvalidArgument(
            "The input argument of %s compute is empty! Please check.",
            op_name));
    CINNValuePack pack_args = args[0];
    PADDLE_ENFORCE_GE(
        pack_args.size(),
        2U,
        ::common::errors::InvalidArgument("2 input tensors for compute\n"));
    Expr x = pack_args[0];
    Expr index = pack_args[1];
    PADDLE_ENFORCE_NOT_NULL(x.as_tensor(),
                            ::common::errors::InvalidArgument(
                                "Required x must be a tensor. Please check."));
    PADDLE_ENFORCE_NOT_NULL(
        index.as_tensor(),
        ::common::errors::InvalidArgument(
            "Required index must be a tensor. Please check."));
    PADDLE_ENFORCE_NE(
        output_shapes.empty(),
        true,
        ::common::errors::InvalidArgument(
            "The output shape of gather_nd is empty! Please check."));
    auto tensor_x = x.as_tensor_ref();
    auto tensor_index = index.as_tensor_ref();
    VLOG(3) << "x shape: " << utils::Join(tensor_x->shape, ", ")
            << ", index shape: " << utils::Join(tensor_index->shape, ", ")
            << ", output_shapes: " << utils::Join(output_shapes[0], ", ");
    PADDLE_ENFORCE_EQ(pack_args.size(),
                      3U,
                      ::common::errors::InvalidArgument(
                          "The size of pack_args should be 3\n"));
    std::string tensor_name = pack_args[2].operator std::string();
    ir::Tensor out = GatherNd(tensor_x, tensor_index, tensor_name);
    std::vector<CINNValue> res;
    res.push_back(CINNValue(out));
    PADDLE_ENFORCE_NE(
        out_type.empty(),
        true,
        ::common::errors::InvalidArgument(
            "The output type of gather_nd is empty! Please check."));
    *ret = CINNValuePack{res};
  });

  framework::CINNSchedule gather_nd_schedule([=](lang::Args args,
                                                 lang::RetValue *ret) {
    PADDLE_ENFORCE_EQ(!args.empty(),
                      true,
                      ::common::errors::InvalidArgument(
                          "The input argument of gather_nd_schedule "
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
        ::common::errors::InvalidArgument(
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
          [&](std::variant<common::UnknownArch, common::ARMArch>) {
            CINN_NOT_IMPLEMENTED;
          },
          [&](common::X86Arch) {
            pe::IRScheduleInjectiveCPU(
                ir_sch, output_shapes.front(), target, true);
          },
          [&](common::NVGPUArch) {
            pe::IRGpuScheduleInjective(ir_sch, output_shapes.front(), target);
          },
          [&](std::variant<common::HygonDCUArchHIP, common::HygonDCUArchSYCL>) {
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

  framework::CINNCompute gather_nd_compute([=](lang::Args args,
                                               lang::RetValue *ret) {
    PADDLE_ENFORCE_NE(
        args.empty(),
        true,
        ::common::errors::InvalidArgument(
            "The input argument of %s compute is empty! Please check.",
            op_name));
    CINNValuePack pack_args = args[0];
    PADDLE_ENFORCE_GE(
        pack_args.size(),
        2U,
        ::common::errors::InvalidArgument("2 input tensors for compute\n"));
    Expr x = pack_args[0];
    Expr index = pack_args[1];
    PADDLE_ENFORCE_NOT_NULL(x.as_tensor(),
                            ::common::errors::InvalidArgument(
                                "Required x must be a tensor. Please check."));
    PADDLE_ENFORCE_NOT_NULL(
        index.as_tensor(),
        ::common::errors::InvalidArgument(
            "Required index must be a tensor. Please check."));
    PADDLE_ENFORCE_NE(
        output_shapes.empty(),
        true,
        ::common::errors::InvalidArgument(
            "The output shape of gather_nd is empty! Please check."));
    auto tensor_x = x.as_tensor_ref();
    auto tensor_index = index.as_tensor_ref();
    VLOG(3) << "x shape: " << utils::Join(tensor_x->shape, ", ")
            << ", index shape: " << utils::Join(tensor_index->shape, ", ")
            << ", output_shapes: " << utils::Join(output_shapes[0], ", ");
    PADDLE_ENFORCE_EQ(pack_args.size(),
                      3U,
                      ::common::errors::InvalidArgument(
                          "The size of pack_args should be 3\n"));
    std::string tensor_name = pack_args[2].operator std::string();
    ir::Tensor out = GatherNdSymbolic(tensor_x, tensor_index, tensor_name);
    std::vector<CINNValue> res;
    res.push_back(CINNValue(out));
    PADDLE_ENFORCE_NE(
        out_type.empty(),
        true,
        ::common::errors::InvalidArgument(
            "The output type of gather_nd is empty! Please check."));
    *ret = CINNValuePack{res};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(
      gather_nd_compute, lang::PackedFunc(), "strategy.gather_nd.x86", 1);
  return strategy;
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
      .set_attr<cinn::hlir::framework::OpPatternKind>(
          "OpPattern", cinn::hlir::framework::OpPatternKind::kInjective)
      .set_support_level(4);

  return true;
}
