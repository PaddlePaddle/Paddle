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

#include "paddle/cinn/hlir/op/contrib/sort.h"

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
#include "paddle/cinn/ir/tensor.h"
#include "paddle/cinn/lang/compute.h"

namespace cinn {
namespace hlir {
namespace op {

using cinn::common::CINNValue;
using cinn::common::CINNValuePack;

std::vector<ir::Tensor> ArgSort(const ir::Tensor &A,
                                const cinn::common::Target &target,
                                const int &axis,
                                const bool &is_ascend,
                                const std::string &name) {
  std::string find_func_name;
  std::string index_func_name;
  target.arch.Match(
      [&](common::UnknownArch) {
        PADDLE_THROW(::common::errors::Fatal(
            "ArgSort only supports X86 and NVGPU ! Please Check.\n"));
      },
      [&](common::X86Arch) {
        find_func_name.assign("cinn_host_next_smallest_int32");
      },
      [&](common::ARMArch) {
        PADDLE_THROW(::common::errors::Fatal(
            "ArgSort only supports X86 and NVGPU ! Please Check.\n"));
      },
      [&](common::NVGPUArch) {
        find_func_name.assign("cinn_nvgpu_next_smallest_int32");
      },
      [&](common::HygonDCUArchHIP) {
        find_func_name.assign("cinn_hip_next_smallest_int32");
      },
      [&](common::HygonDCUArchSYCL) {
        find_func_name.assign("cinn_sycl_next_smallest_int32");
      });
  if (is_ascend) {
    index_func_name =
        cinn::hlir::GetExternFuncName(target, A->type(), "lt_num");
  } else {
    index_func_name =
        cinn::hlir::GetExternFuncName(target, A->type(), "gt_num");
  }
  int pos_axis = axis;
  if (pos_axis < 0) {
    pos_axis += A->shape.size();
  }
  auto positions = Compute(
      A->shape,
      [=](const std::vector<Expr> &indices) {
        Expr offset(0);
        Expr stride(1);
        for (int i = 0; i < indices.size(); i++) {
          if (i < pos_axis) {
            offset = offset * A->shape[i] + indices[i];
          } else if (i == pos_axis) {
            offset = offset * A->shape[i];
          } else {
            offset = offset * A->shape[i] + indices[i];
            stride = stride * A->shape[i];
          }
        }
        offset = optim::ArithSimplify(offset);
        stride = optim::ArithSimplify(stride);
        auto A_shape_axis = A->shape[pos_axis];
        return lang::CallExtern(index_func_name,
                                {A, A_shape_axis, A(indices), offset, stride});
      },
      name + "_temp");
  auto res = Compute(
      A->shape,
      [=](const std::vector<Expr> &indices) {
        Expr offset(0);
        Expr stride(1);
        for (int i = 0; i < indices.size(); i++) {
          if (i < pos_axis) {
            offset = offset * A->shape[i] + indices[i];
          } else if (i == pos_axis) {
            offset = offset * A->shape[i];
          } else {
            offset = offset * A->shape[i] + indices[i];
            stride = stride * A->shape[i];
          }
        }
        offset = optim::ArithSimplify(offset);
        stride = optim::ArithSimplify(stride);

        auto A_shape_axis = A->shape[pos_axis];
        auto idx = lang::CallExtern(
            find_func_name,
            {positions, A_shape_axis, indices[pos_axis], offset, stride});
        return idx;
      },
      name);
  return {res, positions};
}

std::vector<ir::Tensor> Sort(const ir::Tensor &A,
                             const cinn::common::Target &target,
                             const int &axis,
                             const bool &is_ascend,
                             const std::string &name) {
  int pos_axis = axis;
  if (pos_axis < 0) {
    pos_axis += A->shape.size();
  }
  auto sort_index = ArgSort(A, target, pos_axis, is_ascend, name + "_index");
  auto res = Compute(
      A->shape,
      [=](const std::vector<Expr> &indices) {
        std::vector<Expr> A_indices(indices);
        A_indices[pos_axis] = sort_index.at(0)(indices);
        return A(A_indices);
      },
      name);
  return {res, sort_index.at(0), sort_index.at(1)};
}

std::shared_ptr<framework::OpStrategy> StrategyForSort(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<int>> &output_shapes,
    const Target &target) {
  auto attr_store = attrs.attr_store;
  std::string op_name("sort");

  PADDLE_ENFORCE_GE(
      attr_store.count("axis"),
      1,
      ::common::errors::InvalidArgument(
          "The attr_store doesn't have the attribute of 'axis'."));
  int axis = absl::get<int>(attr_store.at("axis"));
  bool is_ascend = true;
  if (attr_store.count("is_ascend")) {
    is_ascend = absl::get<bool>(attr_store.at("is_ascend"));
  }

  framework::CINNCompute sort_compute([=](lang::Args args,
                                          lang::RetValue *ret) {
    PADDLE_ENFORCE_NE(
        args.empty(),
        true,
        ::common::errors::InvalidArgument(
            "The input argument of Sort compute is empty! Please check."));
    CINNValuePack pack_args = args[0];
    PADDLE_ENFORCE_GE(pack_args.size(),
                      1U,
                      ::common::errors::InvalidArgument(
                          "At least 1 input tensors for Sort compute\n"));
    Expr A = pack_args[0];
    PADDLE_ENFORCE_NOT_NULL(
        A.as_tensor(),
        ::common::errors::InvalidArgument(
            "Required Input must be a tensor. Please check."));
    PADDLE_ENFORCE_NE(output_shapes.empty(),
                      true,
                      ::common::errors::InvalidArgument(
                          "The output shape of Sort is empty! Please check."));
    auto tensor_A = A.as_tensor_ref();
    VLOG(3) << "A shape: " << utils::Join(tensor_A->shape, ", ")
            << ", output_shapes: " << utils::Join(output_shapes[0], ", ");
    PADDLE_ENFORCE_EQ(pack_args.size(),
                      2U,
                      ::common::errors::InvalidArgument(
                          "The input argument's size of Sort should be 2"));
    PADDLE_ENFORCE_EQ(
        pack_args[1].is_string(),
        true,
        ::common::errors::InvalidArgument(
            "Required pack_args[1] must be a string. Please check."));
    std::string tensor_name = pack_args[1].operator std::string();
    std::vector<ir::Tensor> out =
        Sort(tensor_A, target, axis, is_ascend, tensor_name);

    std::vector<CINNValue> res{
        CINNValue(out[0]), CINNValue(out[1]), CINNValue(out[2])};
    PADDLE_ENFORCE_NE(out_type.empty(),
                      true,
                      ::common::errors::InvalidArgument(
                          "The output type of Sort is empty! Please check."));

    *ret = CINNValuePack{res};
  });

  framework::CINNSchedule sort_schedule([=](lang::Args args,
                                            lang::RetValue *ret) {
    PADDLE_ENFORCE_NE(
        args.empty(),
        true,
        ::common::errors::InvalidArgument("The input argument of sort_schedule "
                                          "compute is empty! Please check."));

    cinn::common::CINNValuePack arg_pack = args[0];
    std::vector<Expr> vec_ast;
    for (int i = 0; i < arg_pack.size(); i++) {
      if (arg_pack[i].is_expr()) {
        Expr temp = arg_pack[i];
        vec_ast.emplace_back(temp);
      }
    }
    PADDLE_ENFORCE_NE(
        vec_ast.empty(),
        true,
        ::common::errors::InvalidArgument(
            "The vec_ast of sort_schedule compute is empty! Please check."));
    ir::ModuleExpr mod_expr(vec_ast);
    ir::IRSchedule ir_sch(mod_expr);
    ir_sch.MergeExprs();
    auto blocks = ir_sch.GetAllBlocks();
    // TODO(Shixiaowei02): remove external calls, do not use local
    // variables, because the size will exceed the limit.
    ir_sch.SetBuffer(blocks[0], "local");
    ir_sch.SetBuffer(blocks[1], "local");

    int64_t prod_size = std::accumulate(output_shapes[0].begin(),
                                        output_shapes[0].end(),
                                        1,
                                        std::multiplies<int>());
    if (prod_size > 1 && std::holds_alternative<common::X86Arch>(target.arch)) {
      pe::IRScheduleInjectiveCPU(ir_sch, output_shapes.front(), target, true);
    }
    std::vector<cinn::common::CINNValue> res{
        cinn::common::CINNValue(ir_sch.GetModule().GetExprs().at(0))};
    *ret = cinn::common::CINNValuePack{res};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(sort_compute, sort_schedule, "strategy.sort", 1);
  return strategy;
}

std::shared_ptr<framework::OpStrategy> StrategyForArgSort(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<int>> &output_shapes,
    const Target &target) {
  auto attr_store = attrs.attr_store;
  PADDLE_ENFORCE_GE(
      attr_store.count("axis"),
      1,
      ::common::errors::InvalidArgument(
          "The attr_store doesn't have the attribute of 'axis'."));
  int axis = absl::get<int>(attr_store.at("axis"));
  bool is_ascend = true;
  if (attr_store.count("is_ascend")) {
    is_ascend = absl::get<bool>(attr_store.at("is_ascend"));
  }

  framework::CINNCompute argsort_compute([=](lang::Args args,
                                             lang::RetValue *ret) {
    PADDLE_ENFORCE_NE(
        args.empty(),
        true,
        ::common::errors::InvalidArgument(
            "The input argument of Argsort compute is empty! Please check."));
    CINNValuePack pack_args = args[0];
    PADDLE_ENFORCE_GE(pack_args.size(),
                      1U,
                      ::common::errors::InvalidArgument(
                          "The input arguments' size of ArgSort should be 1"));
    Expr A = pack_args[0];
    PADDLE_ENFORCE_NOT_NULL(
        A.as_tensor(),
        ::common::errors::InvalidArgument(
            "Required Input must be a tensor. Please check."));
    PADDLE_ENFORCE_NE(
        output_shapes.empty(),
        true,
        ::common::errors::InvalidArgument(
            "The output shape of Argsort is empty! Please check."));
    auto tensor_A = A.as_tensor_ref();

    VLOG(3) << "A shape: " << utils::Join(tensor_A->shape, ", ")
            << ", output_shapes: " << utils::Join(output_shapes[0], ", ");
    PADDLE_ENFORCE_EQ(pack_args.size(),
                      3U,
                      ::common::errors::InvalidArgument(
                          "The input argument's size of ArgSort should be 3"));
    PADDLE_ENFORCE_EQ(
        pack_args[1].is_string(),
        true,
        ::common::errors::InvalidArgument(
            "Required pack_args[1] must be a string. Please check."));
    std::string tensor_name = pack_args[1].operator std::string();
    auto out = ArgSort(tensor_A, target, axis, is_ascend, tensor_name);
    std::vector<CINNValue> res;
    res.push_back(CINNValue(out.at(0)));
    res.push_back(CINNValue(out.at(1)));
    PADDLE_ENFORCE_NE(
        out_type.empty(),
        true,
        ::common::errors::InvalidArgument(
            "The output type of ArgSort is empty! Please check."));
    *ret = CINNValuePack{res};
  });

  framework::CINNSchedule argsort_schedule([=](lang::Args args,
                                               lang::RetValue *ret) {
    PADDLE_ENFORCE_NE(args.empty(),
                      true,
                      ::common::errors::InvalidArgument(
                          "The input argument of argsort_schedule "
                          "compute is empty! Please check."));
    cinn::common::CINNValuePack arg_pack = args[0];
    std::vector<Expr> vec_ast;
    for (int i = 0; i < arg_pack.size(); i++) {
      if (arg_pack[i].is_expr()) {
        Expr temp = arg_pack[i];
        vec_ast.emplace_back(temp);
      }
    }
    PADDLE_ENFORCE_NE(
        vec_ast.empty(),
        true,
        ::common::errors::InvalidArgument(
            "The vec_ast of argsort_schedule compute is empty! Please check."));
    ir::ModuleExpr mod_expr(vec_ast);
    ir::IRSchedule ir_sch(mod_expr);
    ir_sch.MergeExprs();
    auto blocks = ir_sch.GetAllBlocks();
    // TODO(Shixiaowei02): remove external calls, do not use local variables,
    // because the size will exceed the limit.
    // TODO(lanxianghit): There is a bug, setting buffer to "local" here will
    // cause the var declared twice at CodeGen. ir_sch.SetBuffer(blocks[0],
    // "local");
    int64_t prod_size = std::accumulate(output_shapes[0].begin(),
                                        output_shapes[0].end(),
                                        1,
                                        std::multiplies<int>());
    if (prod_size > 1 && std::holds_alternative<common::X86Arch>(target.arch)) {
      pe::IRScheduleInjectiveCPU(ir_sch, output_shapes.front(), target, true);
    }
    std::vector<cinn::common::CINNValue> res{
        cinn::common::CINNValue(ir_sch.GetModule().GetExprs().at(0))};
    *ret = cinn::common::CINNValuePack{res};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(argsort_compute, argsort_schedule, "strategy.argsort", 1);
  return strategy;
}

}  // namespace op
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(sort_ops) {
  CINN_REGISTER_OP(sort)
      .describe(
          "Sort a variable x along the given axis and return sorted Variable.")
      .set_num_inputs(1)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>(
          "CINNStrategy", cinn::hlir::op::StrategyForSort)
      .set_attr<cinn::hlir::framework::OpPatternKind>(
          "OpPattern", cinn::hlir::framework::OpPatternKind::kNonFusible)
      .set_support_level(4);

  CINN_REGISTER_OP(argsort)
      .describe("Sort a variable x along the given axis and return indices.")
      .set_num_inputs(1)
      .set_num_outputs(2)
      .set_attr<cinn::hlir::framework::StrategyFunction>(
          "CINNStrategy", cinn::hlir::op::StrategyForArgSort)
      .set_attr<cinn::hlir::framework::OpPatternKind>(
          "OpPattern", cinn::hlir::framework::OpPatternKind::kNonFusible)
      .set_support_level(4);

  CINN_REGISTER_OP(top_k)
      .describe(
          "Find values and indices of the k largest entries for the last "
          "dimension..")
      .set_num_inputs(1)
      .set_num_outputs(2)
      .set_attr<cinn::hlir::framework::OpPatternKind>(
          "OpPattern", cinn::hlir::framework::OpPatternKind::kNonFusible)
      .set_support_level(4);

  return true;
}
