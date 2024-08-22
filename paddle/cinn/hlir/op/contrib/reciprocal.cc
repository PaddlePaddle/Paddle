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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "paddle/cinn/common/cas.h"
#include "paddle/cinn/common/common.h"
#include "paddle/cinn/common/context.h"
#include "paddle/cinn/common/macros.h"
#include "paddle/cinn/common/target.h"
#include "paddle/cinn/hlir/framework/op.h"
#include "paddle/cinn/hlir/framework/op_strategy.h"
#include "paddle/cinn/hlir/op/contrib/reciprocal.h"
#include "paddle/cinn/hlir/op/op_util.h"
#include "paddle/cinn/hlir/pe/ir_schedule_pe.h"
#include "paddle/cinn/hlir/pe/nn.h"
#include "paddle/cinn/hlir/pe/schedule.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/ir/tensor.h"
#include "paddle/cinn/lang/compute.h"
#include "paddle/common/flags.h"

namespace cinn {
namespace hlir {
namespace op {

using cinn::common::_CINNValuePack_;
using cinn::common::CINNValue;
using cinn::common::CINNValuePack;
using framework::OpStrategy;
using framework::shape_t;
using framework::StrategyFunction;

ir::Tensor Reciprocal(const ir::Tensor &input, const std::string &output_name) {
  std::string extern_func = "cinn_";

  extern_func += "reciprocal";

  if (input->type().is_float(32)) {
    extern_func += "_fp32";
  } else if (input->type().is_float(64)) {
    extern_func += "_fp64";
  } else if (input->type().is_bfloat16()) {
    extern_func += "_bf16";
  } else if (input->type().is_float16()) {
    extern_func += "_fp16";
  } else {
    CINN_NOT_IMPLEMENTED
  }

  return {Compute(
      input->shape,
      [=](const std::vector<Expr> &indice) {
        ir::Tensor out_tensor(input);
        auto e = out_tensor(indice);
        return cinn::common::make_const(input->type(), 1.0f) / e;
      },
      output_name)};
}

std::shared_ptr<OpStrategy> StrategyForReciprocal(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<int>> &output_shapes,
    const Target &target) {
  std::string op_name("reciprocal");

  framework::CINNCompute reciprocal_compute(
      [=](lang::Args args, lang::RetValue *ret) {
        PADDLE_ENFORCE_NE(
            args.empty(),
            true,
            ::common::errors::InvalidArgument(
                "The input argument of %s compute is empty! Please check.",
                op_name));
        CINNValuePack pack_args = args[0];
        PADDLE_ENFORCE_NE(
            pack_args.empty(),
            true,
            ::common::errors::InvalidArgument(
                "At least one input tensor for %s compute.", op_name));
        PADDLE_ENFORCE_EQ(pack_args.size(),
                          2,
                          ::common::errors::InvalidArgument(
                              "The input argument's size of reciprocal op "
                              "should be 2."));
        PADDLE_ENFORCE_EQ(
            pack_args[1].is_string(),
            true,
            ::common::errors::InvalidArgument(
                "Required pack_args[1] must be a string. Please check."));
        std::string tensor_name = pack_args[1].operator std::string();

        Expr A = pack_args[0];
        PADDLE_ENFORCE_NOT_NULL(
            A.as_tensor(),
            ::common::errors::InvalidArgument(
                "Required Input must be a tensor. Please check."));
        PADDLE_ENFORCE_NE(
            output_shapes.empty(),
            true,
            ::common::errors::InvalidArgument(
                "The output shape of reciprocal is empty! Please check."));
        auto tensor_A = A.as_tensor_ref();
        VLOG(3) << "A shape: " << utils::Join(tensor_A->shape, ", ")
                << ", output_shapes: " << utils::Join(output_shapes[0], ", ");

        PADDLE_ENFORCE_EQ(pack_args.size(),
                          2U,
                          ::common::errors::InvalidArgument(
                              "The input argument's size of reciprocal op "
                              "should be 2."));

        tensor_name = pack_args[1].operator std::string();

        ir::Tensor out = Reciprocal(tensor_A, tensor_name);
        std::vector<CINNValue> res;
        res.push_back(CINNValue(out));
        PADDLE_ENFORCE_NE(
            out_type.empty(),
            true,
            ::common::errors::InvalidArgument(
                "The output type of Reciprocal is empty! Please check."));
        *ret = CINNValuePack{res};
      });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(reciprocal_compute,
                    GetInjectiveScheduleFunc(output_shapes, target),
                    "strategy.reciprocal.x86",
                    1);
  return strategy;
}

std::shared_ptr<OpStrategy> StrategyForReciprocalSymbolic(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<ir::Dim>> &output_shapes,
    const Target &target) {
  std::string op_name("reciprocal");

  framework::CINNCompute reciprocal_compute([=](lang::Args args,
                                                lang::RetValue *ret) {
    PADDLE_ENFORCE_NE(
        args.empty(),
        true,
        ::common::errors::InvalidArgument(
            "The input argument of %s compute is empty! Please check.",
            op_name));
    CINNValuePack pack_args = args[0];
    PADDLE_ENFORCE_NE(
        pack_args.empty(),
        true,
        ::common::errors::InvalidArgument(
            "At least one input tensor for %s compute.", op_name));
    PADDLE_ENFORCE_EQ(pack_args.size(),
                      2,
                      ::common::errors::InvalidArgument(
                          "The input argument's size of reciprocal op "
                          "should be 2."));
    PADDLE_ENFORCE_EQ(
        pack_args[1].is_string(),
        true,
        ::common::errors::InvalidArgument(
            "Required pack_args[1] must be a string. Please check."));
    std::string tensor_name = pack_args[1].operator std::string();

    Expr A = pack_args[0];
    PADDLE_ENFORCE_NOT_NULL(
        A.as_tensor(),
        ::common::errors::InvalidArgument(
            "Required Input must be a tensor. Please check."));
    PADDLE_ENFORCE_NE(
        output_shapes.empty(),
        true,
        ::common::errors::InvalidArgument(
            "The output shape of reciprocal_compute is empty! Please check."));
    auto tensor_A = A.as_tensor_ref();
    VLOG(3) << "A shape: " << utils::Join(tensor_A->shape, ", ")
            << ", output_shapes: " << utils::Join(output_shapes[0], ", ");
    PADDLE_ENFORCE_EQ(pack_args.size(),
                      2U,
                      ::common::errors::InvalidArgument(
                          "The input argument's size of reciprocal op "
                          "should be 2."));
    tensor_name = pack_args[1].operator std::string();

    ir::Tensor out = Reciprocal(tensor_A, tensor_name);
    std::vector<CINNValue> res;
    res.push_back(CINNValue(out));
    PADDLE_ENFORCE_NE(
        out_type.empty(),
        true,
        ::common::errors::InvalidArgument(
            "The output type of Reciprocal is empty! Please check."));
    *ret = CINNValuePack{res};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(
      reciprocal_compute, lang::PackedFunc(), "strategy.reciprocal.x86", 1);
  return strategy;
}

}  // namespace op
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(reciprocal_ops) {
  CINN_REGISTER_OP(reciprocal)
      .describe("Counting Leading Zeros.")
      .set_num_inputs(1)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>(
          "CINNStrategy", cinn::hlir::op::StrategyForReciprocal)
      .set_attr<cinn::hlir::framework::StrategyFunctionSymbolic>(
          "CINNStrategySymbolic", cinn::hlir::op::StrategyForReciprocalSymbolic)
      .set_attr<cinn::hlir::framework::OpPatternKind>(
          "OpPattern", cinn::hlir::framework::OpPatternKind::kElementWise)
      .set_support_level(4);

  return true;
}
