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

#define DEFINE_ACT_GRAD_DEPX_OP_ARGMAP(func_name, op_name, attrs) \
  KernelSignature func_name##GradOpArgumentMapping(               \
      const ArgumentMappingContext& ctx) {                        \
    return KernelSignature(op_name "_grad",                       \
                           {"X", GradVarName("Out")},             \
                           {attrs},                               \
                           {GradVarName("X")});                   \
  }

#define DEFINE_ACT_GRAD_DEPOUT_OP_ARGMAP(func_name, op_name, attrs) \
  KernelSignature func_name##GradOpArgumentMapping(                 \
      const ArgumentMappingContext& ctx) {                          \
    return KernelSignature(op_name "_grad",                         \
                           {"Out", GradVarName("Out")},             \
                           {attrs},                                 \
                           {GradVarName("X")});                     \
  }

#define comma ,

DEFINE_ACT_GRAD_DEPX_OP_ARGMAP(Cos, "cos", );      // NOLINT
DEFINE_ACT_GRAD_DEPX_OP_ARGMAP(Tan, "tan", );      // NOLINT
DEFINE_ACT_GRAD_DEPX_OP_ARGMAP(Acos, "acos", );    // NOLINT
DEFINE_ACT_GRAD_DEPX_OP_ARGMAP(Sin, "sin", );      // NOLINT
DEFINE_ACT_GRAD_DEPX_OP_ARGMAP(Asin, "asin", );    // NOLINT
DEFINE_ACT_GRAD_DEPX_OP_ARGMAP(Atan, "atan", );    // NOLINT
DEFINE_ACT_GRAD_DEPX_OP_ARGMAP(Sinh, "sinh", );    // NOLINT
DEFINE_ACT_GRAD_DEPX_OP_ARGMAP(Cosh, "cosh", );    // NOLINT
DEFINE_ACT_GRAD_DEPX_OP_ARGMAP(Asinh, "asinh", );  // NOLINT
DEFINE_ACT_GRAD_DEPX_OP_ARGMAP(Acosh, "acosh", );  // NOLINT
DEFINE_ACT_GRAD_DEPX_OP_ARGMAP(Atanh, "atanh", );  // NOLINT
DEFINE_ACT_GRAD_DEPX_OP_ARGMAP(BRelu, "brelu", "t_min" comma "t_max");
DEFINE_ACT_GRAD_DEPX_OP_ARGMAP(LeakyRelu, "leaky_relu", "alpha");
DEFINE_ACT_GRAD_DEPX_OP_ARGMAP(ThresholdedRelu,
                               "thresholded_relu",
                               "threshold");
DEFINE_ACT_GRAD_DEPX_OP_ARGMAP(SoftShrink, "soft_shrink", "lambda");
DEFINE_ACT_GRAD_DEPX_OP_ARGMAP(HardShrink, "hard_shrink", "threshold");
DEFINE_ACT_GRAD_DEPX_OP_ARGMAP(TanhShrink, "tanh_shrink", );  // NOLINT
DEFINE_ACT_GRAD_DEPX_OP_ARGMAP(Silu, "silu", );               // NOLINT
DEFINE_ACT_GRAD_DEPX_OP_ARGMAP(LogSigmoid, "logsigmoid", );   // NOLINT
DEFINE_ACT_GRAD_DEPX_OP_ARGMAP(Log, "log", );                 // NOLINT
DEFINE_ACT_GRAD_DEPX_OP_ARGMAP(Log2, "log2", );               // NOLINT
DEFINE_ACT_GRAD_DEPX_OP_ARGMAP(Log10, "log10", );             // NOLINT
DEFINE_ACT_GRAD_DEPX_OP_ARGMAP(Log1p, "log1p", );             // NOLINT

DEFINE_ACT_GRAD_DEPOUT_OP_ARGMAP(Relu, "relu", );        // NOLINT
DEFINE_ACT_GRAD_DEPOUT_OP_ARGMAP(Tanh, "tanh", );        // NOLINT
DEFINE_ACT_GRAD_DEPOUT_OP_ARGMAP(Sigmoid, "sigmoid", );  // NOLINT
DEFINE_ACT_GRAD_DEPOUT_OP_ARGMAP(HardSigmoid,
                                 "hard_sigmoid",
                                 "slope" comma "offset");  // NOLINT

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

KernelSignature SigmoidDoubleGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature(
      "sigmoid_double_grad", {"Out", "DDX", "DOut"}, {}, {"DOutNew", "DDOut"});
}

KernelSignature SigmoidTripleGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature("sigmoid_triple_grad",
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

KernelSignature EluOpArgumentMapping(const ArgumentMappingContext& ctx) {
  return KernelSignature("elu", {"X"}, {"alpha"}, {"Out"});
}

KernelSignature EluGradOpArgumentMapping(const ArgumentMappingContext& ctx) {
  return KernelSignature("elu_grad",
                         {"X", "Out", GradVarName("Out")},
                         {"alpha"},
                         {GradVarName("X")});
}

KernelSignature EluDoubleGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature(
      "elu_double_grad", {"X", "DOut", "DDX"}, {"alpha"}, {"DX", "DDOut"});
}

KernelSignature LogDoubleGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature(
      "log_double_grad", {"X", "DOut", "DDX"}, {}, {"DX", "DDOut"});
}

}  // namespace phi

PD_REGISTER_BASE_KERNEL_NAME(relu_grad_grad, relu_double_grad);
PD_REGISTER_BASE_KERNEL_NAME(tanh_grad_grad, tanh_double_grad);
PD_REGISTER_BASE_KERNEL_NAME(leaky_relu_grad_grad, leaky_relu_double_grad);
PD_REGISTER_BASE_KERNEL_NAME(softshrink, soft_shrink);
PD_REGISTER_BASE_KERNEL_NAME(softshrink_grad, soft_shrink_grad);
PD_REGISTER_BASE_KERNEL_NAME(elu_grad_grad, elu_double_grad);
PD_REGISTER_BASE_KERNEL_NAME(sigmoid_grad_grad, sigmoid_double_grad);
PD_REGISTER_BASE_KERNEL_NAME(log_grad_grad, log_double_grad);

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
PD_REGISTER_ARG_MAPPING_FN(softshrink_grad,
                           phi::SoftShrinkGradOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(hard_shrink_grad,
                           phi::HardShrinkGradOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(tanh_shrink_grad,
                           phi::TanhShrinkGradOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(elu, phi::EluOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(elu_grad, phi::EluGradOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(elu_grad_grad, phi::EluDoubleGradOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(silu_grad, phi::SiluGradOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(sigmoid_grad, phi::SigmoidGradOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(sigmoid_grad_grad,
                           phi::SigmoidDoubleGradOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(sigmoid_triple_grad,
                           phi::SigmoidTripleGradOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(logsigmoid_grad,
                           phi::LogSigmoidGradOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(hard_sigmoid_grad,
                           phi::HardSigmoidGradOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(log_grad, phi::LogGradOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(log_grad_grad, phi::LogDoubleGradOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(log2_grad, phi::Log2GradOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(log10_grad, phi::Log10GradOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(log1p_grad, phi::Log1pGradOpArgumentMapping);
