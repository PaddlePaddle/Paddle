// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/hlir/pe/transform.h"

#include <algorithm>

#include "paddle/cinn/hlir/framework/op.h"
#include "paddle/cinn/hlir/framework/op_strategy.h"
#include "paddle/cinn/hlir/op/op_util.h"
#include "paddle/cinn/hlir/pe/elementwise.h"
#include "paddle/cinn/hlir/pe/ir_schedule_pe.h"
#include "paddle/cinn/hlir/pe/nn.h"
#include "paddle/cinn/hlir/pe/schedule.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/optim/ir_simplify.h"
#include "paddle/cinn/utils/string.h"
#include "paddle/common/enforce.h"
#include "paddle/common/errors.h"
#include "paddle/phi/core/enforce.h"

namespace cinn {
namespace hlir {
namespace op {
using cinn::common::_CINNValuePack_;
using cinn::common::CINNValue;
using cinn::common::CINNValuePack;
using framework::OpStrategy;
using framework::shape_t;
using framework::StrategyFunction;

std::shared_ptr<OpStrategy> StrategyForConcatSymbolic(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<ir::Dim>> &output_shapes,
    const Target &target) {
  framework::CINNCompute concat_compute([=](lang::Args args,
                                            lang::RetValue *ret) {
    PADDLE_ENFORCE_EQ(
        !args.empty(),
        true,
        ::common::errors::InvalidArgument(
            "The input arguments of Concat compute is empty! Please "
            "check."));
    PADDLE_ENFORCE_EQ(!out_type.empty(),
                      true,
                      ::common::errors::InvalidArgument(
                          "Output type of Concat is empty! Please check."));
    CINNValuePack pack_args = args[0];
    int input_size = pack_args.size() - 1;
    PADDLE_ENFORCE_GE(input_size,
                      1UL,
                      ::common::errors::InvalidArgument(
                          "the num of input tensors for Concat compute should "
                          "be greater than or equal to 2, but got %d.",
                          input_size));
    PADDLE_ENFORCE_EQ(
        !output_shapes.empty(),
        true,
        ::common::errors::InvalidArgument(
            "The output_shapes of Concat is empty! Please check."));
    int axis = 0;
    if (attrs.attr_store.count("axis")) {
      axis = absl::get<int>(attrs.attr_store.at("axis"));
    }

    std::vector<ir::Tensor> input_tensors;
    for (int i = 0; i < input_size; i++) {
      Expr tensor = pack_args[i];
      PADDLE_ENFORCE(
          tensor.as_tensor(),
          ::common::errors::InvalidArgument(
              "The pack_args[%d] should be tensor! Please check.", i));
      input_tensors.push_back(tensor.as_tensor_ref());
    }

    PADDLE_ENFORCE_EQ(
        pack_args[input_size].is_string(),
        true,
        ::common::errors::InvalidArgument(
            "The pack_args[%d] should be string! Please check.", input_size));
    std::string tensor_name = pack_args[input_size].operator std::string();

    auto out = pe::Concat(input_tensors, axis, tensor_name);

    *ret = CINNValuePack(std::vector<CINNValue>({CINNValue(out)}));
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(concat_compute, "strategy.concat.x86", 1);
  return strategy;
}

std::shared_ptr<OpStrategy> StrategyForReverseSymbolic(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<ir::Dim>> &output_shapes,
    const Target &target) {
  // check output shape
  PADDLE_ENFORCE_EQ(!output_shapes.empty(),
                    true,
                    ::common::errors::InvalidArgument(
                        "Output shape is empty! Please check."));
  // get axis[0, n_dim)
  std::vector<int> axis;
  if (attrs.attr_store.find("axis") != attrs.attr_store.end()) {
    axis = absl::get<std::vector<int>>(attrs.attr_store.at("axis"));
    for (auto &e : axis) {
      if (e >= static_cast<int>(output_shapes[0].size()) ||
          e < -1 * static_cast<int>(output_shapes[0].size())) {
        PADDLE_THROW(::common::errors::InvalidArgument(
            "axis is not in [0, n_dim), Please check."));
      }
      if (e < 0) {
        e += output_shapes[0].size();
      }
    }
  }

  framework::CINNCompute reverse_compute(
      [=](lang::Args args, lang::RetValue *ret) {
        PADDLE_ENFORCE_EQ(!args.empty(),
                          true,
                          ::common::errors::InvalidArgument(
                              "The input argument of reverse compute is empty! "
                              "Please check."));
        CINNValuePack input_args = args[0];
        PADDLE_ENFORCE_EQ(!input_args.empty(),
                          true,
                          ::common::errors::InvalidArgument(
                              "at least one input tensor for reverse compute"));
        Expr A = input_args[0];
        PADDLE_ENFORCE(A.as_tensor(),
                       ::common::errors::InvalidArgument(
                           "The input_args[0] should be a tensor! Please "
                           "check."));

        PADDLE_ENFORCE_EQ(input_args.size(),
                          2,
                          ::common::errors::InvalidArgument(
                              "The input number of reverse_sybmbolic "
                              "should be equal to 2, but got %d.",
                              input_args.size()));
        PADDLE_ENFORCE_EQ(input_args[1].is_string(),
                          true,
                          ::common::errors::InvalidArgument(
                              "The 2-th input_args should be a string! Please "
                              "check."));
        std::string tensor_name = input_args[1].operator std::string();
        auto out = pe::Reverse(A.as_tensor_ref(), axis, tensor_name);
        *ret = CINNValuePack{{CINNValue(out)}};
      });

  auto strategy = std::make_shared<framework::OpStrategy>();
  PADDLE_ENFORCE_EQ(out_type.size(),
                    true,
                    ::common::errors::InvalidArgument(
                        "Out_type of reverse op is empty! Please check."));
  strategy->AddImpl(reverse_compute, "strategy.reverse.x86", 1);
  return strategy;
}

std::shared_ptr<OpStrategy> StrategyForTransposeSymbolic(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<ir::Dim>> &output_shapes,
    const Target &target) {
  // check output shape
  PADDLE_ENFORCE_EQ(output_shapes.empty(),
                    false,
                    ::common::errors::InvalidArgument(
                        "Output shape is empty! Please check.\n"));
  PADDLE_ENFORCE_EQ(output_shapes[0].empty(),
                    false,
                    ::common::errors::InvalidArgument(
                        "Output shape is empty! Please check.\n"));

  std::vector<int> axis;
  auto input_shape = inputs[0]->shape;
  if (attrs.attr_store.find("axis") != attrs.attr_store.end()) {
    axis = absl::get<std::vector<int>>(attrs.attr_store.at("axis"));
    PADDLE_ENFORCE_LE(axis.size(),
                      output_shapes[0].size(),
                      ::common::errors::InvalidArgument(
                          "axis size is not equal output_shapes size! Please "
                          "check setting.\n"));
    // check axis and shape
    for (int idx = 0; idx < axis.size(); ++idx) {
      PADDLE_ENFORCE(axis[idx] >= 0 && axis[idx] < axis.size(),
                     ::common::errors::InvalidArgument(
                         "axis is not in the tensor shape."));
      for (int idy = idx + 1; idy < axis.size(); ++idy) {
        PADDLE_ENFORCE_NE(axis[idx],
                          axis[idy],
                          ::common::errors::InvalidArgument(
                              "The same axis parameter exists!"));
      }
    }
  } else {
    PADDLE_THROW(
        ::common::errors::InvalidArgument("axis is not be set! Please check."));
  }

  framework::CINNCompute transpose_compute([=](lang::Args args,
                                               lang::RetValue *ret) {
    PADDLE_ENFORCE(
        !args.empty(),
        ::common::errors::InvalidArgument("The input argument of transpose "
                                          "compute is empty! Please check.\n"));
    CINNValuePack input_args = args[0];
    PADDLE_ENFORCE(!input_args.empty(),
                   ::common::errors::InvalidArgument(
                       "at least one input tensor for transpose compute.\n"));
    Expr A = input_args[0];
    PADDLE_ENFORCE(
        A.as_tensor(),
        ::common::errors::InvalidArgument("The input argument is not Tensor."));
    PADDLE_ENFORCE_EQ(input_args.size(),
                      2,
                      ::common::errors::InvalidArgument(
                          "The input args size must be equal to 2."));
    PADDLE_ENFORCE(
        input_args[1].is_string(),
        ::common::errors::InvalidArgument(
            "The second argument must be of type string and is the name "
            "of the output tensor."));
    std::string tensor_name = input_args[1].operator std::string();

    auto out = pe::Transpose(A.as_tensor_ref(), axis, tensor_name);
    *ret = CINNValuePack{{CINNValue(out)}};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(transpose_compute, "strategy.transpose.x86", 1);
  return strategy;
}

std::shared_ptr<OpStrategy> StrategyForGatherSymbolic(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<ir::Dim>> &output_shapes,
    const Target &target) {
  PADDLE_ENFORCE_NE(output_shapes.size(),
                    0,
                    ::common::errors::InvalidArgument(
                        "The shape of output is empty! Please check again."));
  PADDLE_ENFORCE_NE(output_shapes[0].size(),
                    0,
                    ::common::errors::InvalidArgument(
                        "The shape of output is empty! Please check again."));

  VLOG(4) << "The output passed in StrategyForGather: "
          << utils::Join(output_shapes[0], ", ");
  PADDLE_ENFORCE_NE(
      out_type.size(),
      0,
      ::common::errors::InvalidArgument(
          "The output type of Gather is empty! Please check again."));

  int axis = 0;
  if (attrs.attr_store.contains("axis")) {
    axis = absl::get<int>(attrs.attr_store.at("axis"));
  }
  axis = axis < 0 ? axis + static_cast<int>(inputs[0]->shape.size()) : axis;

  std::vector<Expr> output_shape = ToCinnExprs(output_shapes[0]);

  framework::CINNCompute gather_compute{
      [axis, output_shape = std::move(output_shape)](lang::Args args,
                                                     lang::RetValue *ret) {
        VLOG(4) << "The axis value used in gather_compute: " << axis;
        PADDLE_ENFORCE_NE(args.size(),
                          0,
                          ::common::errors::InvalidArgument(
                              "The input args are empty! Please check again."));
        CINNValuePack input_args = args[0];
        int input_size = input_args.size();
        PADDLE_ENFORCE_GE(input_size,
                          2,
                          ::common::errors::InvalidArgument(
                              "Require 2 input tensors for Gather compute."));
        Expr x = input_args[0];
        PADDLE_ENFORCE_NE(x.as_tensor(),
                          nullptr,
                          ::common::errors::InvalidArgument(
                              "The first input args's type should be Tensor"));
        Expr index = input_args[1];
        PADDLE_ENFORCE_NE(index.as_tensor(),
                          nullptr,
                          ::common::errors::InvalidArgument(
                              "The first input args's type should be Tensor"));

        std::string tensor_name = input_args[2].operator std::string();

        auto out = pe::Gather(x.as_tensor_ref(),
                              index.as_tensor_ref(),
                              axis,
                              output_shape,
                              tensor_name);
        std::vector<CINNValue> res{CINNValue(out)};
        *ret = CINNValuePack{res};
      }};

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(gather_compute, "strategy.gather.x86", 1);
  return strategy;
}

template <typename T = int>
std::vector<T> GetIntVectorFromAttr(const utils::Attribute &attr) {
  if (absl::holds_alternative<std::vector<int64_t>>(attr)) {
    const auto &attr_data = absl::get<std::vector<int64_t>>(attr);
    return std::vector<T>(attr_data.begin(), attr_data.end());
  } else if (absl::holds_alternative<std::vector<int>>(attr)) {
    const auto &attr_data = absl::get<std::vector<int>>(attr);
    return std::vector<T>(attr_data.begin(), attr_data.end());
  } else if (absl::holds_alternative<bool>(attr)) {
    return std::vector<T>{};
  } else {
    PADDLE_THROW(::common::errors::InvalidArgument(
        "attribute's vector type is invalid!"));
  }
}
std::shared_ptr<OpStrategy> StrategyForSliceSymbolic(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<ir::Dim>> &output_shapes,
    const Target &target) {
  const std::vector<Expr> starts_expr = [&] {
    if (inputs.size() == 3) {
      const auto &value = inputs.at(1).self()->value();
      PADDLE_ENFORCE_EQ(value.has_value(),
                        true,
                        ::common::errors::InvalidArgument(
                            "The inputs.at(1) has no value! Please check."));
      return value.value();
    }
    if (attrs.attr_store.find("starts") != attrs.attr_store.end()) {
      return ToCinnExprs(
          GetIntVectorFromAttr<int64_t>(attrs.attr_store.at("starts")));
    } else {
      PADDLE_THROW(::common::errors::InvalidArgument(
          "The Slice op doesn't find [starts] attribute!"));
    }
  }();
  const std::vector<Expr> ends_expr = [&] {
    if (inputs.size() == 3) {
      const auto &value = inputs.at(2).self()->value();
      PADDLE_ENFORCE_EQ(value.has_value(),
                        true,
                        ::common::errors::InvalidArgument(
                            "The inputs.at(2) has no value! Please check."));
      return value.value();
    }
    if (attrs.attr_store.find("ends") != attrs.attr_store.end()) {
      return ToCinnExprs(
          GetIntVectorFromAttr<int64_t>(attrs.attr_store.at("ends")));
    } else {
      PADDLE_THROW(::common::errors::InvalidArgument(
          "The Slice op doesn't find [ends] attribute!"));
    }
  }();
  const std::vector<int> axes = [&] {
    std::vector<int> axes;
    if (attrs.attr_store.find("axes") != attrs.attr_store.end()) {
      axes = GetIntVectorFromAttr(attrs.attr_store.at("axes"));
    }
    if (axes.empty()) {
      for (int i = 0; i < starts_expr.size(); i++) {
        axes.push_back(i);
      }
    }
    return axes;
  }();
  const std::vector<Expr> strides_expr = [&] {
    std::vector<int> strides;
    if (attrs.attr_store.find("strides") != attrs.attr_store.end()) {
      strides = GetIntVectorFromAttr(attrs.attr_store.at("strides"));
    }
    if (strides.empty()) {
      for (int i = 0; i < starts_expr.size(); i++) {
        strides.push_back(1);
      }
    }
    return ToCinnExprs(strides);
  }();
  const std::vector<int> decrease_axis = [&] {
    if (attrs.attr_store.find("decrease_axis") != attrs.attr_store.end()) {
      return GetIntVectorFromAttr(attrs.attr_store.at("decrease_axis"));
    }
    return std::vector<int>{};
  }();

  PADDLE_ENFORCE_EQ(!starts_expr.empty(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The Slice op doesn't find [starts] attribute!"
                        "It is a mandatory attribute, please check."));
  PADDLE_ENFORCE_EQ(!ends_expr.empty(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The Slice op doesn't find [ends] attribute!"
                        "It is a mandatory attribute, please check."));
  PADDLE_ENFORCE_EQ(
      starts_expr.size(),
      ends_expr.size(),
      ::common::errors::InvalidArgument(
          "The size of [starts] and [ends] must be identical! But the size of "
          "[starts] is %d, the size of [ends] is %d.",
          starts_expr.size(),
          ends_expr.size()));
  PADDLE_ENFORCE_EQ(
      starts_expr.size(),
      axes.size(),
      ::common::errors::InvalidArgument(
          "The size of [starts] and [axes] must be identical! But the size of "
          "[starts] is %d, the size of [axes] is %d.",
          starts_expr.size(),
          axes.size()));
  PADDLE_ENFORCE_EQ(
      starts_expr.size(),
      strides_expr.size(),
      ::common::errors::InvalidArgument(
          "The size of [starts] and [strides] must be identical! But the "
          "size of [starts] is %d, the size of [strides] is %d.",
          starts_expr.size(),
          strides_expr.size()));

  std::vector<Expr> output_shape;
  for (auto &i : output_shapes[0]) {
    output_shape.push_back(i->dim_expr);
    PADDLE_ENFORCE_EQ(
        output_shape.back().type().valid(),
        true,
        ::common::errors::InvalidArgument(
            "The output_shapes[0] has invalid type! Please check."));
  }

  framework::CINNCompute slice_compute([=](lang::Args args,
                                           lang::RetValue *ret) {
    PADDLE_ENFORCE_EQ(!args.empty(),
                      true,
                      ::common::errors::InvalidArgument(
                          "The input arguments of slice compute is empty! "
                          "Please check."));
    CINNValuePack arg_pack = args[0];
    PADDLE_ENFORCE_EQ(!arg_pack.empty(),
                      true,
                      ::common::errors::InvalidArgument(
                          "The input tensors of slice compute is empty! "
                          "Please check."));
    Expr A_expr = arg_pack[0];
    PADDLE_ENFORCE(A_expr.as_tensor(),
                   ::common::errors::InvalidArgument(
                       "The 1-th args_packs should be a tensor! Please "
                       "check."));
    ir::Tensor A = A_expr.as_tensor_ref();

    const std::string tensor_name = [&] {
      if (arg_pack.size() == 2 || arg_pack.size() == 4) {
        PADDLE_ENFORCE_EQ(arg_pack.back().is_string(),
                          true,
                          ::common::errors::InvalidArgument(
                              "The last input_args should be a string when "
                              "the size of arg_pack is 2 or 4! Please check."
                              "The size of arg_pack is %d.",
                              arg_pack.size()));
        return arg_pack.back().operator std::string();
      }
      PADDLE_THROW(::common::errors::InvalidArgument(
          "The slice op doesn't find output tensor name! The size of "
          "arg_pack is %d.",
          arg_pack.size()));
    }();

    auto out = pe::SliceSymbolic(A,
                                 starts_expr,
                                 axes,
                                 strides_expr,
                                 decrease_axis,
                                 output_shape,
                                 tensor_name);
    VLOG(4) << "out: " << out;
    *ret = CINNValuePack{{CINNValue(out)}};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(slice_compute, "strategy.slice.x86", 1);

  return strategy;
}

}  // namespace op
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(transform_ops) {
  CINN_REGISTER_OP(concat)
      .describe(
          "This operator is used to concat two input tensors X and Y on "
          "specified axis.")
      .set_num_inputs(2)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunctionSymbolic>(
          "CINNStrategySymbolic", cinn::hlir::op::StrategyForConcatSymbolic)
      .set_attr<cinn::hlir::framework::OpPatternKind>(
          "OpPattern", cinn::hlir::framework::OpPatternKind::kInjective)
      .set_support_level(4);

  CINN_REGISTER_OP(reverse)
      .describe("This operator implements the meta op reverse.")
      .set_num_inputs(1)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunctionSymbolic>(
          "CINNStrategySymbolic", cinn::hlir::op::StrategyForReverseSymbolic)
      .set_attr<cinn::hlir::framework::OpPatternKind>(
          "OpPattern", cinn::hlir::framework::OpPatternKind::kInjective)
      .set_support_level(4);

  CINN_REGISTER_OP(transpose)
      .describe("This operator implements the meta op transpose.")
      .set_num_inputs(1)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunctionSymbolic>(
          "CINNStrategySymbolic", cinn::hlir::op::StrategyForTransposeSymbolic)
      .set_attr<cinn::hlir::framework::OpPatternKind>(
          "OpPattern", cinn::hlir::framework::OpPatternKind::kInjective)
      .set_support_level(4);

  CINN_REGISTER_OP(slice)
      .describe("This operator implements the slice layer")
      .set_num_inputs(1)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunctionSymbolic>(
          "CINNStrategySymbolic", cinn::hlir::op::StrategyForSliceSymbolic)
      .set_attr<cinn::hlir::framework::OpPatternKind>(
          "OpPattern", cinn::hlir::framework::OpPatternKind::kInjective)
      .set_support_level(4);

  CINN_REGISTER_OP(gather)
      .describe(
          "This operator is used to create a new tensor which indexes the "
          "`input` tensor along dimension `axis` using "
          "the entries in `index`.")
      .set_num_inputs(2)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunctionSymbolic>(
          "CINNStrategySymbolic", cinn::hlir::op::StrategyForGatherSymbolic)
      .set_attr<cinn::hlir::framework::OpPatternKind>(
          "OpPattern", cinn::hlir::framework::OpPatternKind::kInjective)
      .set_support_level(4);

  return true;
}
