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

#include "paddle/cinn/hlir/op/contrib/resize.h"

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
#include "paddle/cinn/hlir/pe/elementwise.h"
#include "paddle/cinn/hlir/pe/ir_schedule_pe.h"
#include "paddle/cinn/hlir/pe/transform.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/tensor.h"
#include "paddle/cinn/lang/builtin.h"
#include "paddle/cinn/lang/compute.h"

namespace cinn {
namespace hlir {
namespace op {

using common::CINNValuePack;

#define __get_pixel(input, h, w, n, c, y, x)                         \
  input({n,                                                          \
         c,                                                          \
         common::AutoSimplify(                                       \
             ir::Max::Make(ir::Min::Make(y, h - Expr(1)), Expr(0))), \
         common::AutoSimplify(                                       \
             ir::Max::Make(ir::Min::Make(x, w - Expr(1)), Expr(0)))})

ir::Tensor Resize(const ir::Tensor &input,
                  const common::Target &target,
                  const std::vector<int> &out_shape,
                  const std::string &mode,
                  const std::string &output_name) {
  std::string func_name;

  if (target.arch == common::Target::Arch::NVGPU) {
    func_name.assign("cinn_cuda_resize_");
  } else if (target.arch == common::Target::Arch::X86) {
    func_name.assign("cinn_host_resize_");
  } else {
    LOG(FATAL) << "Resize only supports X86 and NVGPU ! Please Check.\n";
  }

  if (mode == "bilinear") {
    func_name.append("bilinear");
  } else if (mode == "bicubic") {
    func_name.append("bicubic");
  }

  Expr in_h = input->shape[2];
  Expr in_w = input->shape[3];
  Expr out_h = Expr(out_shape[0]);
  Expr out_w = Expr(out_shape[1]);

  std::vector<Expr> new_shape = {
      input->shape[0], input->shape[1], out_h, out_w};
  ir::Tensor res = lang::Compute(
      {new_shape},
      [=](const std::vector<Expr> &indices) {
        Expr out_y = indices[2];
        Expr out_x = indices[3];
        Expr value;

        if (mode == "nearest") {
          Expr in_y = ir::Cast::Make(common::F32(), in_h) /
                      ir::Cast::Make(common::F32(), out_h) *
                      ir::Cast::Make(common::F32(), out_y);
          Expr in_x = ir::Cast::Make(common::F32(), in_w) /
                      ir::Cast::Make(common::F32(), out_w) *
                      ir::Cast::Make(common::F32(), out_x);
          Expr in_y_int = ir::Cast::Make(common::Int(32), lang::Floor(in_y));
          Expr in_x_int = ir::Cast::Make(common::Int(32), lang::Floor(in_x));
          std::vector<Expr> in_indices = {
              indices[0], indices[1], in_y_int, in_x_int};
          value = input(in_indices);

        } else if (mode == "bilinear") {
          value = lang::CallExtern(func_name,
                                   {input,
                                    input->shape[1],
                                    in_h,
                                    in_w,
                                    out_h,
                                    out_w,
                                    indices[0],
                                    indices[1],
                                    out_y,
                                    out_x});

        } else if (mode == "bicubic") {
          value = lang::CallExtern(func_name,
                                   {input,
                                    input->shape[1],
                                    in_h,
                                    in_w,
                                    out_h,
                                    out_w,
                                    indices[0],
                                    indices[1],
                                    out_y,
                                    out_x});
        }

        return value;
      },
      common::UniqName(output_name));

  return res;
}

std::vector<std::vector<int>> InferShapeForResize(
    const std::vector<std::vector<int>> &inputs_shape,
    const framework::AttrMapType &attrs) {
  CHECK_EQ(inputs_shape[0].size(), 4U)
      << "The input's shape size should be 4! Please check again.";

  CHECK(attrs.find("out_shape") != attrs.end())
      << "Cannot find \"out_shape\" attribute in \"resize\" op, Please Check.";
  std::vector<int> out_shape;
  out_shape = absl::get<std::vector<int>>(attrs.at("out_shape"));
  CHECK_EQ(out_shape.size(), 2U) << "The length of out_shape must be 2.";
  CHECK(out_shape[0] > 0 && out_shape[1] > 0)
      << "The element of out_shape must be great that 0.";

  CHECK(attrs.find("mode") != attrs.end())
      << "Cannot find \"mode\" attribute in \"resize\" op, Please Check.";
  std::string mode = absl::get<std::string>(attrs.at("mode"));
  CHECK(mode == "nearest" || mode == "bilinear" || mode == "bicubic")
      << "Resize only supports `nearest`, `bilinear` and `bicubic` mode.";

  framework::shape_t x_shape = inputs_shape[0];
  std::vector<int> new_shape;
  new_shape.push_back(x_shape[0]);
  new_shape.push_back(x_shape[1]);
  new_shape.push_back(out_shape[0]);
  new_shape.push_back(out_shape[1]);

  return {new_shape};
}

std::vector<Type> InferDtypeForResize(const std::vector<Type> &inputs_type,
                                      const framework::AttrMapType &attrs) {
  CHECK(!inputs_type.empty())
      << "The input's type size is 0! Please check again.";
  CHECK(inputs_type[0] == Int(32)) << "Resize only supports int32 type input.";
  std::vector<Type> res{inputs_type[0]};
  return res;
}

std::shared_ptr<framework::OpStrategy> StrategyForResize(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<int>> &output_shapes,
    const Target &target) {
  std::vector<int> out_shape;
  std::string mode = "bilinear";

  for (auto &iter : attrs.attr_store) {
    if (iter.first == "out_shape") {
      out_shape = absl::get<std::vector<int>>(iter.second);
    } else if (iter.first == "mode") {
      mode = absl::get<std::string>(iter.second);
    }
  }

  CHECK(mode == "nearest" || mode == "bilinear" || mode == "bicubic")
      << "Resize only supports `nearest`, `bilinear` and `bicubic` mode.";

  framework::CINNCompute resize_compute([=](lang::Args args,
                                            lang::RetValue *ret) {
    CHECK(!args.empty())
        << "The input arguments of Resize compute is empty! Please check.\n";
    CINNValuePack pack_args = args[0];
    CHECK_GE(pack_args.size(), 1U)
        << "at least 1 input tensors for Resize compute\n";
    Expr A = pack_args[0];
    CHECK(A.as_tensor());
    CHECK(!output_shapes.empty());
    auto tensor_A = A.as_tensor_ref();
    VLOG(3) << "A shape: " << utils::Join(tensor_A->shape, ", ")
            << ", output_shapes: " << utils::Join(output_shapes[0], ", ");

    CHECK_EQ(pack_args.size(), 2U);
    std::string tensor_name = pack_args[1].operator std::string();

    ir::Tensor out = Resize(tensor_A, target, out_shape, mode, tensor_name);

    std::vector<common::CINNValue> res;
    auto stages = CreateStages({tensor_A});
    stages->InsertLazily(out);
    res.push_back(common::CINNValue(out));
    res.push_back(common::CINNValue(stages));
    *ret = common::CINNValuePack{res};
  });

  framework::CINNSchedule resize_schedule([=](lang::Args args,
                                              lang::RetValue *ret) {
    CHECK(!args.empty())
        << "The input argument of resize schedule is empty! Please check.\n";
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
    int64_t prod_size = std::accumulate(output_shapes[0].begin(),
                                        output_shapes[0].end(),
                                        1,
                                        std::multiplies<int>());
    if (prod_size > 1) {
      if (target.arch == Target::Arch::NVGPU) {
        pe::IRCudaScheduleInjective(ir_sch, output_shapes.front(), target);
      } else if (target.arch == Target::Arch::X86) {
        pe::IRScheduleInjectiveCPU(ir_sch, output_shapes.front(), target, true);
      }
    }
    std::vector<common::CINNValue> res{
        common::CINNValue(ir_sch.GetModule().GetExprs().at(0))};
    *ret = common::CINNValuePack{res};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(resize_compute, resize_schedule, "strategy.resize.x86", 1);

  return strategy;
}

}  // namespace op
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(resize_ops) {
  CINN_REGISTER_OP(resize)
      .describe(" ")
      .set_num_inputs(1)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>(
          "CINNStrategy", cinn::hlir::op::StrategyForResize)
      .set_attr("infershape",
                MakeOpFunction(cinn::hlir::op::InferShapeForResize))
      .set_attr("inferdtype",
                MakeOpFunction(cinn::hlir::op::InferDtypeForResize))
      .set_attr<cinn::hlir::framework::OpPatternKind>(
          "OpPattern", cinn::hlir::framework::OpPatternKind::kNonFusible)
      .set_support_level(4);

  return true;
}
