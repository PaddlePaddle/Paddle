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

KernelSignature ElementwiseSubDoubleGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature(
      "subtract_double_grad", {"Y", "DDX", "DDY", "DOut"}, {"axis"}, {"DDOut"});
}

KernelSignature ElementwiseDivGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature("divide_grad",
                         {"X", "Y", "Out", GradVarName("Out")},
                         {"axis"},
                         {GradVarName("X"), GradVarName("Y")});
}

KernelSignature ElementwiseFMinGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature("elementwise_fmin_grad",
                         {"X", "Y", GradVarName("Out")},
                         {"axis"},
                         {GradVarName("X"), GradVarName("Y")});
}

KernelSignature ElementwiseDivDoubleGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature("divide_double_grad",
                         {"Y", "Out", "DX", "DDX", "DDY"},
                         {"axis"},
                         {GradVarName("Y"), "DOut", "DDOut"});
}

KernelSignature ElementwiseMulGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature("multiply_grad",
                         {"X", "Y", GradVarName("Out")},
                         {"axis"},
                         {GradVarName("X"), GradVarName("Y")});
}

KernelSignature ElementwiseFMaxGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature("elementwise_fmax_grad",
                         {"X", "Y", GradVarName("Out")},
                         {"axis"},
                         {GradVarName("X"), GradVarName("Y")});
}

KernelSignature ElementwiseMulDoubleGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature("multiply_double_grad",
                         {"X", "Y", "DOut", "DDX", "DDY"},
                         {"axis"},
                         {GradVarName("X"), GradVarName("Y"), "DDOut"});
}

KernelSignature ElementwiseMulTripleGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature(
      "multiply_triple_grad",
      {"X", "Y", "DOut", "DDX", "DDY", "D_DX", "D_DY", "D_DDOut"},
      {"axis"},
      {"D_X", "D_Y", "D_DOut", "D_DDX", "D_DDY"});
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
PD_REGISTER_BASE_KERNEL_NAME(elementwise_sub_grad_grad, subtract_double_grad);
PD_REGISTER_BASE_KERNEL_NAME(elementwise_div_grad, divide_grad);
PD_REGISTER_BASE_KERNEL_NAME(elementwise_div_grad_grad, divide_double_grad);
PD_REGISTER_BASE_KERNEL_NAME(elementwise_mul_grad, multiply_grad);
PD_REGISTER_BASE_KERNEL_NAME(elementwise_mul_grad_grad, multiply_double_grad);
PD_REGISTER_BASE_KERNEL_NAME(elementwise_mul_triple_grad, multiply_triple_grad);

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
PD_REGISTER_ARG_MAPPING_FN(elementwise_sub_grad_grad,
                           phi::ElementwiseSubDoubleGradOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(elementwise_div_grad,
                           phi::ElementwiseDivGradOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(elementwise_div_grad_grad,
                           phi::ElementwiseDivDoubleGradOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(elementwise_mul_grad,
                           phi::ElementwiseMulGradOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(elementwise_mul_grad_grad,
                           phi::ElementwiseMulDoubleGradOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(elementwise_mul_triple_grad,
                           phi::ElementwiseMulTripleGradOpArgumentMapping);

PD_REGISTER_ARG_MAPPING_FN(elementwise_fmax_grad,
                           phi::ElementwiseFMaxGradOpArgumentMapping);

PD_REGISTER_ARG_MAPPING_FN(elementwise_fmin_grad,
                           phi::ElementwiseFMinGradOpArgumentMapping);
