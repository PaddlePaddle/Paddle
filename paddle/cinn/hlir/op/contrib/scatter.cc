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

#include "cinn/hlir/op/contrib/scatter.h"

#include <gflags/gflags.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "cinn/common/cas.h"
#include "cinn/common/common.h"
#include "cinn/common/context.h"
#include "cinn/common/macros.h"
#include "cinn/hlir/framework/node.h"
#include "cinn/hlir/framework/op.h"
#include "cinn/hlir/framework/op_strategy.h"
#include "cinn/hlir/pe/elementwise.h"
#include "cinn/hlir/pe/ir_schedule_pe.h"
#include "cinn/hlir/pe/nn.h"
#include "cinn/hlir/pe/schedule.h"
#include "cinn/hlir/pe/transform.h"
#include "cinn/ir/ir.h"
#include "cinn/ir/ir_base.h"
#include "cinn/ir/tensor.h"
#include "cinn/lang/builtin.h"
#include "cinn/lang/compute.h"

DECLARE_bool(cinn_ir_schedule);

namespace cinn {
namespace hlir {
namespace op {

using common::CINNValue;
using common::CINNValuePack;

ir::Tensor Scatter(const ir::Tensor &A,
                   const ir::Tensor &B,
                   const ir::Tensor &C,
                   const common::Target &target,
                   const int &axis,
                   const std::string &name) {
  CHECK_EQ(A->shape.size(), B->shape.size());
  CHECK_EQ(A->shape.size(), C->shape.size());

  std::string extern_fun_name;
  if (target.arch == common::Target::Arch::NVGPU) {
    extern_fun_name.assign("cinn_cuda_find_int_nd");
  } else if (target.arch == common::Target::Arch::X86) {
    extern_fun_name.assign("cinn_host_find_int_nd");
  } else {
    LOG(FATAL) << "Scatter only support X86 and NVGPU ! Please Check.\n";
  }

  int pos_axis = axis;
  if (pos_axis < 0) {
    pos_axis += C->shape.size();
  }

  ir::Tensor transpose_B;
  if (pos_axis == A->shape.size() - 1) {
    transpose_B = B;
  } else {
    std::vector<int> new_axes;
    for (int i = 0; i < A->shape.size(); ++i) {
      if (i != pos_axis) {
        new_axes.push_back(i);
      }
    }
    new_axes.push_back(pos_axis);
    transpose_B = pe::Transpose(B, new_axes, B->name + "_index_transpose");
  }
  auto res = Compute(
      C->shape,
      [=](const std::vector<Expr> &indices) {
        Expr offset(0);
        for (int i = 0; i < indices.size(); i++) {
          if (i != pos_axis) {
            offset = offset * C->shape[i] + indices[i];
          }
        }
        auto B_shape_axis = B->shape[pos_axis];
        offset = common::AutoSimplify(offset * B_shape_axis);
        auto idx = lang::CallExtern(
            extern_fun_name,
            {transpose_B, B_shape_axis, indices[pos_axis], offset, Expr(1)});
        std::vector<Expr> A_indices(indices);
        A_indices[pos_axis] = idx;
        auto keep = ir::EQ::Make(idx, Expr(-1));
        return ir::Select::Make(keep, C(indices), A(A_indices));
      },
      name);
  return res;
}

ir::Tensor ScatterNd(const ir::Tensor &A,
                     const ir::Tensor &B,
                     const ir::Tensor &C,
                     const common::Target &target,
                     const std::vector<int> &axes,
                     const std::string &name) {
  CHECK(!A->shape.empty());
  CHECK_EQ(A->shape.size() + 1, B->shape.size());
  CHECK_EQ(A->shape.size() + axes.size() - 1, C->shape.size());

  std::string extern_fun_name;
  if (target.arch == common::Target::Arch::NVGPU) {
    extern_fun_name.assign("cinn_cuda_find_int_nd");
  } else if (target.arch == common::Target::Arch::X86) {
    extern_fun_name.assign("cinn_host_find_int_nd");
  } else {
    LOG(FATAL) << "ScatterNd only support X86 and NVGPU ! Please Check.\n";
  }

  std::vector<int> pos_axes;
  for (auto axis : axes) {
    if (axis < 0) {
      pos_axes.push_back(axis + C->shape.size());
    } else {
      pos_axes.push_back(axis);
    }
  }

  auto res = Compute(
      C->shape,
      [=](const std::vector<Expr> &indices) {
        auto offset = Expr(0);
        std::vector<Expr> A_indices;
        for (int i = 0; i < indices.size(); i++) {
          if (std::find(pos_axes.begin(), pos_axes.end(), i) ==
              pos_axes.end()) {
            offset = offset * C->shape[i] + indices[i];
            A_indices.push_back(indices[i]);
          }
        }
        offset = offset * B->shape[B->shape.size() - 2] *
                 B->shape[B->shape.size() - 1];
        auto keep = Expr(true);
        std::vector<Expr> idx;
        for (int i = 0; i < pos_axes.size(); ++i) {
          auto cur_idx =
              lang::CallExtern(extern_fun_name,
                               {B,
                                B->shape[B->shape.size() - 2],
                                indices[pos_axes[i]],
                                common::AutoSimplify(offset + Expr(i)),
                                Expr(static_cast<int>(pos_axes.size()))});
          if (idx.empty()) {
            idx.push_back(cur_idx);
            A_indices.push_back(cur_idx);
          } else {
            keep = ir::And::Make(keep, ir::EQ::Make(idx[0], cur_idx));
            idx[0] = cur_idx;
          }
        }
        keep = common::AutoSimplify(
            ir::And::Make(keep, ir::EQ::Make(idx[0], Expr(-1))));
        return ir::Select::Make(keep, C(indices), A(A_indices));
      },
      name);
  return res;
}

std::shared_ptr<framework::OpStrategy> StrategyForScatter(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<int>> &output_shapes,
    const Target &target) {
  auto attr_store = attrs.attr_store;
  CHECK(attr_store.count("axis")) << "find no attr of axis";
  int axis = absl::get<int>(attr_store.at("axis"));
  std::string op_name("scatter");

  framework::CINNCompute scatter_compute(
      [=](lang::Args args, lang::RetValue *ret) {
        CHECK(!args.empty()) << "The input arguments of " << op_name
                             << " compute is empty! Please check.\n";
        CINNValuePack pack_args = args[0];
        CHECK_GE(pack_args.size(), 3U)
            << "3 input tensors for " << op_name << " compute\n";
        Expr A = pack_args[0];
        Expr B = pack_args[1];
        Expr C = pack_args[2];
        CHECK(A.as_tensor());
        CHECK(B.as_tensor());
        CHECK(C.as_tensor());
        CHECK(!output_shapes.empty());
        auto tensor_A = A.as_tensor_ref();
        auto tensor_B = B.as_tensor_ref();
        auto tensor_C = C.as_tensor_ref();
        auto stages = CreateStages({tensor_A, tensor_B, tensor_C});
        VLOG(3) << "A shape: " << utils::Join(tensor_A->shape, ", ")
                << ", B shape: " << utils::Join(tensor_B->shape, ", ")
                << ", output_shapes: " << utils::Join(output_shapes[0], ", ");
        std::string tensor_name = UniqName("Scatter_out");
        if (FLAGS_cinn_ir_schedule) {
          CHECK_EQ(pack_args.size(), 4U);
          tensor_name = pack_args[3].operator std::string();
        }
        ir::Tensor out =
            Scatter(tensor_A, tensor_B, tensor_C, target, axis, tensor_name);
        std::vector<CINNValue> res;
        stages->InsertLazily(out);
        res.push_back(CINNValue(out));
        CHECK(!out_type.empty())
            << "Output type of " << op_name << " is empty! Please check.\n";
        res.push_back(CINNValue(stages));
        *ret = CINNValuePack{res};
      });

  framework::CINNSchedule scatter_schedule([=](lang::Args args,
                                               lang::RetValue *ret) {
    if (FLAGS_cinn_ir_schedule) {
      CHECK(!args.empty())
          << "The input argument of scatter_schedule is empty! Please check.\n";
      common::CINNValuePack arg_pack = args[0];
      std::vector<Expr> vec_ast;
      for (int i = 0; i < arg_pack.size(); i++) {
        if (arg_pack[i].is_expr()) {
          Expr temp = arg_pack[i];
          vec_ast.emplace_back(temp);
        }
      }
      CHECK(!vec_ast.empty());
      ir::ModuleExpr mod_expr(vec_ast);
      ir::IRSchedule ir_sch(mod_expr);
      ir_sch.MergeExprs();
      long prod_size = std::accumulate(output_shapes[0].begin(),
                                       output_shapes[0].end(),
                                       1,
                                       std::multiplies<int>());
      if (prod_size > 1) {
        pe::IRInjectiveSchedule(ir_sch, output_shapes.front(), target);
      }
      std::vector<common::CINNValue> res{
          common::CINNValue(ir_sch.GetModule().GetExprs().at(0))};
      *ret = common::CINNValuePack{res};
    } else {
      CHECK(!args.empty())
          << "The input argument of scatter_schedule is empty! Please check.\n";
      CINNValuePack arg_pack = args[0];
      Expr out = arg_pack[0];
      CHECK(out.as_tensor());
      *ret = arg_pack;
    }
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(
      scatter_compute, scatter_schedule, "strategy.scatter.x86", 1);
  return strategy;
}

std::shared_ptr<framework::OpStrategy> StrategyForScatterNd(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<int>> &output_shapes,
    const Target &target) {
  auto attr_store = attrs.attr_store;
  CHECK(attr_store.count("axes")) << "find no attr of axis";
  std::vector<int> axes = absl::get<std::vector<int>>(attr_store.at("axes"));
  std::string op_name("scatter_nd");

  framework::CINNCompute scatter_nd_compute(
      [=](lang::Args args, lang::RetValue *ret) {
        CHECK(!args.empty()) << "The input arguments of " << op_name
                             << " compute is empty! Please check.\n";
        CINNValuePack pack_args = args[0];
        CHECK_GE(pack_args.size(), 3U)
            << "3 input tensors for " << op_name << " compute\n";
        Expr A = pack_args[0];
        Expr B = pack_args[1];
        Expr C = pack_args[2];
        CHECK(A.as_tensor());
        CHECK(B.as_tensor());
        CHECK(C.as_tensor());
        CHECK(!output_shapes.empty());
        auto tensor_A = A.as_tensor_ref();
        auto tensor_B = B.as_tensor_ref();
        auto tensor_C = C.as_tensor_ref();
        auto stages = CreateStages({tensor_A, tensor_B, tensor_C});
        VLOG(3) << "A shape: " << utils::Join(tensor_A->shape, ", ")
                << ", B shape: " << utils::Join(tensor_B->shape, ", ")
                << ", output_shapes: " << utils::Join(output_shapes[0], ", ");
        std::string tensor_name = UniqName("ScatterNd_out");
        if (FLAGS_cinn_ir_schedule) {
          CHECK_EQ(pack_args.size(), 4U);
          tensor_name = pack_args[3].operator std::string();
        }
        ir::Tensor out =
            ScatterNd(tensor_A, tensor_B, tensor_C, target, axes, tensor_name);
        std::vector<CINNValue> res;
        stages->InsertLazily(out);
        res.push_back(CINNValue(out));
        CHECK(!out_type.empty())
            << "Output type of " << op_name << " is empty! Please check.\n";
        res.push_back(CINNValue(stages));
        *ret = CINNValuePack{res};
      });

  framework::CINNSchedule scatter_nd_schedule(
      [=](lang::Args args, lang::RetValue *ret) {
        if (FLAGS_cinn_ir_schedule) {
          CHECK(!args.empty()) << "The input argument of scatter_nd_schedule "
                                  "is empty! Please check.\n";
          common::CINNValuePack arg_pack = args[0];
          std::vector<Expr> vec_ast;
          for (int i = 0; i < arg_pack.size(); i++) {
            if (arg_pack[i].is_expr()) {
              Expr temp = arg_pack[i];
              vec_ast.emplace_back(temp);
            }
          }
          CHECK(!vec_ast.empty());
          ir::ModuleExpr mod_expr(vec_ast);
          ir::IRSchedule ir_sch(mod_expr);
          ir_sch.MergeExprs();
          long prod_size = std::accumulate(output_shapes[0].begin(),
                                           output_shapes[0].end(),
                                           1,
                                           std::multiplies<int>());
          if (prod_size > 1) {
            pe::IRInjectiveSchedule(ir_sch, output_shapes.front(), target);
          }
          std::vector<common::CINNValue> res{
              common::CINNValue(ir_sch.GetModule().GetExprs().at(0))};
          *ret = common::CINNValuePack{res};
        } else {
          CHECK(!args.empty()) << "The input argument of scatter_nd_schedule "
                                  "is empty! Please check.\n";
          CINNValuePack arg_pack = args[0];
          Expr out = arg_pack[0];
          CHECK(out.as_tensor());
          *ret = arg_pack;
        }
      });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(
      scatter_nd_compute, scatter_nd_schedule, "strategy.scatter_nd.x86", 1);
  return strategy;
}

std::vector<std::vector<int>> InferShapeForScatter(
    const std::vector<std::vector<int>> &inputs_shape,
    const framework::AttrMapType &attrs) {
  CHECK_EQ(inputs_shape.size(), 3U)
      << "The input's shape size should be 3! Please check again.";
  std::vector<std::vector<int>> res{inputs_shape[2]};
  return res;
}

std::vector<Type> InferDtypeForScatter(const std::vector<Type> &inputs_type,
                                       const framework::AttrMapType &attrs) {
  CHECK_EQ(inputs_type.size(), 3U)
      << "The input's type size should be 3! Please check again.";
  CHECK_EQ(inputs_type[1], Int(32))
      << "The index's type should be int! Please check again.";
  std::vector<Type> res{inputs_type[2]};
  return res;
}

}  // namespace op
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(scatter_ops) {
  CINN_REGISTER_OP(scatter)
      .describe("Scatter.")
      .set_num_inputs(3)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>(
          "CINNStrategy", cinn::hlir::op::StrategyForScatter)
      .set_attr("infershape",
                MakeOpFunction(cinn::hlir::op::InferShapeForScatter))
      .set_attr("inferdtype",
                MakeOpFunction(cinn::hlir::op::InferDtypeForScatter))
      .set_attr<cinn::hlir::framework::OpPatternKind>(
          "OpPattern", cinn::hlir::framework::OpPatternKind::kInjective)
      .set_support_level(4);

  CINN_REGISTER_OP(scatter_nd)
      .describe("ScatterNd.")
      .set_num_inputs(3)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>(
          "CINNStrategy", cinn::hlir::op::StrategyForScatterNd)
      .set_attr("infershape",
                MakeOpFunction(cinn::hlir::op::InferShapeForScatter))
      .set_attr("inferdtype",
                MakeOpFunction(cinn::hlir::op::InferDtypeForScatter))
      .set_support_level(4);

  return true;
}
