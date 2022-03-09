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

#define DefineActGradDepXOpArgMap(func_name, op_name, attrs) \
  KernelSignature func_name##GradOpArgumentMapping(          \
      const ArgumentMappingContext& ctx) {                   \
    return KernelSignature(op_name "_grad",                  \
                           {"X", GradVarName("Out")},        \
                           attrs,                            \
                           {GradVarName("X")});              \
  }

#define DefineActGradDepOutOpArgMap(func_name, op_name, attrs) \
  KernelSignature func_name##GradOpArgumentMapping(            \
      const ArgumentMappingContext& ctx) {                     \
    return KernelSignature(op_name "_grad",                    \
                           {"Out", GradVarName("Out")},        \
                           attrs,                              \
                           {GradVarName("X")});                \
  }

#define comma ,

DefineActGradDepXOpArgMap(Cos, "cos", {});
DefineActGradDepXOpArgMap(Tan, "tan", {});
DefineActGradDepXOpArgMap(Acos, "acos", {});
DefineActGradDepXOpArgMap(Sin, "sin", {});
DefineActGradDepXOpArgMap(Asin, "asin", {});
DefineActGradDepXOpArgMap(Atan, "atan", {});
DefineActGradDepXOpArgMap(Sinh, "sinh", {});
DefineActGradDepXOpArgMap(Cosh, "cosh", {});
DefineActGradDepXOpArgMap(Asinh, "asinh", {});
DefineActGradDepXOpArgMap(Acosh, "acosh", {});
DefineActGradDepXOpArgMap(Atanh, "atanh", {});
DefineActGradDepXOpArgMap(BRelu, "brelu", {"t_min" comma "t_max"});
DefineActGradDepXOpArgMap(LeakyRelu, "leaky_relu", {"alpha"});
DefineActGradDepXOpArgMap(ThresholdedRelu, "thresholded_relu", {"threshold"});

DefineActGradDepOutOpArgMap(Relu, "relu", {});
DefineActGradDepOutOpArgMap(Tanh, "tanh", {});

KernelSignature ReluDoubleGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature("relu_double_grad", {"Out", "DDX"}, {}, {"DDOut"});
}

KernelSignature TanhDoubleGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature(
      "tanh_double_grad", {"Out", "DDX", "DOut"}, {}, {"DOutNew", "DDOut"});
}

KernelSignature TanhTripleGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature("tanh_triple_grad",
                         {"Out", "DDX", "DOut", "D_DDOut", "D_DOut_New"},
                         {},
                         {"D_OutNew", "D_DOut", "D_DDx"});
}

KernelSignature LeakyReluDoubleGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature(
      "leaky_relu_double_grad", {"X", "DDX"}, {"alpha"}, {"DDOut"});
}

KernelSignature LeakyReluOpArgumentMapping(const ArgumentMappingContext& ctx) {
  return KernelSignature("leaky_relu", {"X"}, {"alpha"}, {"Out"});
}

}  // namespace phi

PD_REGISTER_BASE_KERNEL_NAME(relu_grad_grad, relu_double_grad);
PD_REGISTER_BASE_KERNEL_NAME(tanh_grad_grad, tanh_double_grad);
PD_REGISTER_BASE_KERNEL_NAME(leaky_relu_grad_grad, leaky_relu_double_grad);

PD_REGISTER_ARG_MAPPING_FN(cos_grad, phi::CosGradOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(tan_grad, phi::TanGradOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(acos_grad, phi::AcosGradOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(sin_grad, phi::SinGradOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(asin_grad, phi::AsinGradOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(atan_grad, phi::AtanGradOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(sinh_grad, phi::SinhGradOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(cosh_grad, phi::CoshGradOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(asinh_grad, phi::AsinhGradOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(acosh_grad, phi::AcoshGradOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(atanh_grad, phi::AtanhGradOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(relu_grad, phi::ReluGradOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(relu_grad_grad,
                           phi::ReluDoubleGradOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(tanh_grad, phi::TanhGradOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(tanh_grad_grad,
                           phi::TanhDoubleGradOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(tanh_triple_grad,
                           phi::TanhTripleGradOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(brelu_grad, phi::BReluGradOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(leaky_relu, phi::LeakyReluOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(leaky_relu_grad,
                           phi::LeakyReluGradOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(leaky_relu_grad_grad,
                           phi::LeakyReluDoubleGradOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(thresholded_relu_grad,
                           phi::ThresholdedReluGradOpArgumentMapping);
