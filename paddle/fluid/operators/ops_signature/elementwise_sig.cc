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
  if (axis == -1) {
    return KernelSignature("add", {"X", "Y"}, {}, {"Out"});
  }
  return KernelSignature("add_raw", {"X", "Y"}, {"axis"}, {"Out"});
}

KernelSignature ElementwiseGradAddOpArgumentMapping(
    const ArgumentMappingContext& ctx UNUSED) {
  return KernelSignature("grad_add", {"X", "Y"}, {}, {"Out"});
}

KernelSignature ElementwiseSubOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  int axis = paddle::any_cast<int>(ctx.Attr("axis"));
  if (axis == -1) {
    return KernelSignature("subtract", {"X", "Y"}, {}, {"Out"});
  }
  return KernelSignature("subtract_raw", {"X", "Y"}, {"axis"}, {"Out"});
}

KernelSignature ElementwiseMulOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  int axis = paddle::any_cast<int>(ctx.Attr("axis"));
  if (ctx.IsDenseTensorInput("X")) {
    if (axis == -1) {
      return KernelSignature("multiply", {"X", "Y"}, {}, {"Out"});
    }
    return KernelSignature("multiply_raw", {"X", "Y"}, {"axis"}, {"Out"});
  } else {
    if (axis == -1) {
      return KernelSignature("multiply_sr", {"X", "Y"}, {}, {"Out"});
    }
    return KernelSignature("multiply_raw_sr", {"X", "Y"}, {"axis"}, {"Out"});
  }
}

KernelSignature ElementwiseDivOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  int axis = paddle::any_cast<int>(ctx.Attr("axis"));
  if (axis == -1) {
    return KernelSignature("divide", {"X", "Y"}, {}, {"Out"});
  }
  return KernelSignature("divide_raw", {"X", "Y"}, {"axis"}, {"Out"});
}

KernelSignature ElementwiseMaxOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  if (ctx.IsForInferShape()) {
    return KernelSignature("maximum_raw", {"X", "Y"}, {"axis"}, {"Out"});
  }
  int axis = paddle::any_cast<int>(ctx.Attr("axis"));
  if (axis == -1) {
    return KernelSignature("maximum", {"X", "Y"}, {}, {"Out"});
  }
  return KernelSignature("maximum_raw", {"X", "Y"}, {"axis"}, {"Out"});
}

KernelSignature ElementwiseMinOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  if (ctx.IsForInferShape()) {
    return KernelSignature("minimum_raw", {"X", "Y"}, {"axis"}, {"Out"});
  }
  int axis = paddle::any_cast<int>(ctx.Attr("axis"));
  if (axis == -1) {
    return KernelSignature("minimum", {"X", "Y"}, {}, {"Out"});
  }
  return KernelSignature("minimum_raw", {"X", "Y"}, {"axis"}, {"Out"});
}

KernelSignature ElementwiseModOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  if (ctx.IsForInferShape()) {
    return KernelSignature("remainder_raw", {"X", "Y"}, {"axis"}, {"Out"});
  }
  int axis = paddle::any_cast<int>(ctx.Attr("axis"));
  if (axis == -1) {
    return KernelSignature("remainder", {"X", "Y"}, {}, {"Out"});
  }
  return KernelSignature("remainder_raw", {"X", "Y"}, {"axis"}, {"Out"});
}

KernelSignature ElementwiseFloorDivOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  if (ctx.IsForInferShape()) {
    return KernelSignature("floor_divide_raw", {"X", "Y"}, {"axis"}, {"Out"});
  }
  int axis = paddle::any_cast<int>(ctx.Attr("axis"));
  if (axis == -1) {
    return KernelSignature("floor_divide", {"X", "Y"}, {}, {"Out"});
  }
  return KernelSignature("floor_divide_raw", {"X", "Y"}, {"axis"}, {"Out"});
}

KernelSignature ElementwisePowOpArgumentMapping(
    const ArgumentMappingContext& ctx UNUSED) {
  if (ctx.IsForInferShape()) {
    return KernelSignature(
        "elementwise_pow_raw", {"X", "Y"}, {"axis"}, {"Out"});
  }
  int axis = paddle::any_cast<int>(ctx.Attr("axis"));
  if (axis == -1) {
    return KernelSignature("elementwise_pow", {"X", "Y"}, {}, {"Out"});
  }
  return KernelSignature("elementwise_pow_raw", {"X", "Y"}, {"axis"}, {"Out"});
}

KernelSignature ElementwiseAddGradOpArgumentMapping(
    const ArgumentMappingContext& ctx UNUSED) {
  return KernelSignature(
      "add_grad", {"X", "Y", "Out@GRAD"}, {"axis"}, {"X@GRAD", "Y@GRAD"});
}

KernelSignature ElementwiseAddDoubleGradOpArgumentMapping(
    const ArgumentMappingContext& ctx UNUSED) {
  return KernelSignature(
      "add_double_grad", {"Y", "DOut", "DDX", "DDY"}, {"axis"}, {"DDOut"});
}

KernelSignature ElementwiseAddTripleGradOpArgumentMapping(
    const ArgumentMappingContext& ctx UNUSED) {
  return KernelSignature("add_triple_grad",
                         {"DDX", "DDY", "D_DDOut"},
                         {"axis"},
                         {"D_DDX", "D_DDY"});
}

KernelSignature ElementwiseSubGradOpArgumentMapping(
    const ArgumentMappingContext& ctx UNUSED) {
  return KernelSignature(
      "subtract_grad", {"X", "Y", "Out@GRAD"}, {"axis"}, {"X@GRAD", "Y@GRAD"});
}

KernelSignature ElementwiseSubDoubleGradOpArgumentMapping(
    const ArgumentMappingContext& ctx UNUSED) {
  return KernelSignature(
      "subtract_double_grad", {"Y", "DOut", "DDX", "DDY"}, {"axis"}, {"DDOut"});
}

KernelSignature ElementwiseDivGradOpArgumentMapping(
    const ArgumentMappingContext& ctx UNUSED) {
  return KernelSignature("divide_grad",
                         {"X", "Y", "Out", "Out@GRAD"},
                         {"axis"},
                         {"X@GRAD", "Y@GRAD"});
}

KernelSignature ElementwiseDivDoubleGradOpArgumentMapping(
    const ArgumentMappingContext& ctx UNUSED) {
  return KernelSignature("divide_double_grad",
                         {"Y", "Out", "Out@GRAD", "DX", "DDX", "DDY"},
                         {"axis"},
                         {"Y@GRAD", "DOut", "DDOut"});
}

KernelSignature ElementwiseMulGradOpArgumentMapping(
    const ArgumentMappingContext& ctx UNUSED) {
  return KernelSignature(
      "multiply_grad", {"X", "Y", "Out@GRAD"}, {"axis"}, {"X@GRAD", "Y@GRAD"});
}

KernelSignature ElementwiseFMaxOpArgumentMapping(
    const ArgumentMappingContext& ctx UNUSED) {
  return KernelSignature("fmax", {"X", "Y"}, {}, {"Out"});
}

KernelSignature ElementwiseFMinOpArgumentMapping(
    const ArgumentMappingContext& ctx UNUSED) {
  return KernelSignature("fmin", {"X", "Y"}, {}, {"Out"});
}

KernelSignature ElementwiseMulDoubleGradOpArgumentMapping(
    const ArgumentMappingContext& ctx UNUSED) {
  return KernelSignature("multiply_double_grad",
                         {"X", "Y", "DOut", "DDX", "DDY"},
                         {"axis"},
                         {"X@GRAD", "Y@GRAD", "DDOut"});
}

KernelSignature ElementwiseMulTripleGradOpArgumentMapping(
    const ArgumentMappingContext& ctx UNUSED) {
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
PD_REGISTER_BASE_KERNEL_NAME(elementwise_max, maximum);
PD_REGISTER_BASE_KERNEL_NAME(elementwise_min, minimum);
PD_REGISTER_BASE_KERNEL_NAME(elementwise_mod, remainder);
PD_REGISTER_BASE_KERNEL_NAME(elementwise_floordiv, floor_divide);
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
PD_REGISTER_BASE_KERNEL_NAME(elementwise_fmax, fmax);
PD_REGISTER_BASE_KERNEL_NAME(elementwise_fmin, fmin);

PD_REGISTER_ARG_MAPPING_FN(elementwise_add,
                           phi::ElementwiseAddOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(elementwise_sub,
                           phi::ElementwiseSubOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(elementwise_mul,
                           phi::ElementwiseMulOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(elementwise_div,
                           phi::ElementwiseDivOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(elementwise_max,
                           phi::ElementwiseMaxOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(elementwise_min,
                           phi::ElementwiseMinOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(elementwise_mod,
                           phi::ElementwiseModOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(elementwise_floordiv,
                           phi::ElementwiseFloorDivOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(elementwise_pow,
                           phi::ElementwisePowOpArgumentMapping);
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
PD_REGISTER_ARG_MAPPING_FN(elementwise_fmax,
                           phi::ElementwiseFMaxOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(elementwise_fmin,
                           phi::ElementwiseFMinOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(grad_add, phi::ElementwiseGradAddOpArgumentMapping);
