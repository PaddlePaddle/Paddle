/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/core/compat/op_utils.h"

namespace phi {

KernelSignature ElementwiseAddOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  int axis = paddle::any_cast<int>(ctx.Attr("axis"));
  if (ctx.IsDenseTensorInput("X")) {
    if (axis == -1) {
      return KernelSignature("add", {"X", "Y"}, {}, {"Out"});
    }
    return KernelSignature("add_raw", {"X", "Y"}, {"axis"}, {"Out"});
  }
  return KernelSignature("unregistered", {}, {}, {});
}

KernelSignature ElementwiseSubOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  int axis = paddle::any_cast<int>(ctx.Attr("axis"));
  if (ctx.IsDenseTensorInput("X")) {
    if (axis == -1) {
      return KernelSignature("subtract", {"X", "Y"}, {}, {"Out"});
    }
    return KernelSignature("subtract_raw", {"X", "Y"}, {"axis"}, {"Out"});
  }
  return KernelSignature("unregistered", {}, {}, {});
}

KernelSignature ElementwiseMulOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  int axis = paddle::any_cast<int>(ctx.Attr("axis"));
  if (ctx.IsDenseTensorInput("X")) {
    if (axis == -1) {
      return KernelSignature("multiply", {"X", "Y"}, {}, {"Out"});
    }
    return KernelSignature("multiply_raw", {"X", "Y"}, {"axis"}, {"Out"});
  }
  return KernelSignature("unregistered", {}, {}, {});
}

KernelSignature ElementwiseDivOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  int axis = paddle::any_cast<int>(ctx.Attr("axis"));
  if (ctx.IsDenseTensorInput("X")) {
    if (axis == -1) {
      return KernelSignature("divide", {"X", "Y"}, {}, {"Out"});
    }
    return KernelSignature("divide_raw", {"X", "Y"}, {"axis"}, {"Out"});
  }
  return KernelSignature("unregistered", {}, {}, {});
}

KernelSignature ElementwiseAddGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  if (ctx.IsDenseTensorInput("X")) {
    return KernelSignature("add_grad",
                           {"X", "Y", GradVarName("Out")},
                           {"axis"},
                           {GradVarName("X"), GradVarName("Y")});
  }
  return KernelSignature("unregistered", {}, {}, {});
}

KernelSignature ElementwiseAddDoubleGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature(
      "add_double_grad", {"Y", "DDX", "DDY", "DOut"}, {"axis"}, {"DDOut"});
}

KernelSignature ElementwiseAddTripleGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature("add_triple_grad",
                         {"DDX", "DDY", "D_DDOut"},
                         {"axis"},
                         {"D_DDX", "D_DDY"});
}

KernelSignature ElementwiseSubGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  if (ctx.IsDenseTensorInput("X")) {
    return KernelSignature("subtract_grad",
                           {"X", "Y", GradVarName("Out")},
                           {"axis"},
                           {GradVarName("X"), GradVarName("Y")});
  }
  return KernelSignature("unregistered", {}, {}, {});
}

}  // namespace phi

PD_REGISTER_BASE_KERNEL_NAME(elementwise_add, add);
PD_REGISTER_BASE_KERNEL_NAME(elementwise_sub, subtract);
PD_REGISTER_BASE_KERNEL_NAME(elementwise_mul, multiply);
PD_REGISTER_BASE_KERNEL_NAME(elementwise_div, divide);
PD_REGISTER_BASE_KERNEL_NAME(elementwise_add_grad, add_grad);
PD_REGISTER_BASE_KERNEL_NAME(elementwise_add_grad_grad, add_double_grad);
PD_REGISTER_BASE_KERNEL_NAME(elementwise_add_triple_grad, add_triple_grad);
PD_REGISTER_BASE_KERNEL_NAME(elementwise_sub_grad, subtract_grad);

PD_REGISTER_ARG_MAPPING_FN(elementwise_add,
                           phi::ElementwiseAddOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(elementwise_sub,
                           phi::ElementwiseSubOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(elementwise_mul,
                           phi::ElementwiseMulOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(elementwise_div,
                           phi::ElementwiseDivOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(elementwise_add_grad,
                           phi::ElementwiseAddGradOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(elementwise_add_grad_grad,
                           phi::ElementwiseAddDoubleGradOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(elementwise_add_triple_grad,
                           phi::ElementwiseAddTripleGradOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(elementwise_sub_grad,
                           phi::ElementwiseSubGradOpArgumentMapping);
