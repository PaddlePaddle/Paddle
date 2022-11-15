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
    return KernelSignature(                                       \
        op_name "_grad", {"X", "Out@GRAD"}, {attrs}, {"X@GRAD"}); \
  }

#define DEFINE_ACT_GRAD_DEPOUT_OP_ARGMAP(func_name, op_name, attrs) \
  KernelSignature func_name##GradOpArgumentMapping(                 \
      const ArgumentMappingContext& ctx) {                          \
    return KernelSignature(                                         \
        op_name "_grad", {"Out", "Out@GRAD"}, {attrs}, {"X@GRAD"}); \
  }

#define DEFINE_ACT_GRAD_NODEP_OP_ARGMAP(func_name, op_name, attrs) \
  KernelSignature func_name##GradOpArgumentMapping(                \
      const ArgumentMappingContext& ctx) {                         \
    return KernelSignature(                                        \
        op_name "_grad", {"Out@GRAD"}, {attrs}, {"X@GRAD"});       \
  }

#define comma ,

DEFINE_ACT_GRAD_DEPX_OP_ARGMAP(HardTanh, "hard_tanh", "t_min" comma "t_max");
DEFINE_ACT_GRAD_DEPX_OP_ARGMAP(Mish, "mish", "threshold");
DEFINE_ACT_GRAD_DEPX_OP_ARGMAP(HardSwish,
                               "hard_swish",
                               "threshold" comma "scale" comma
                               "offset");                // NOLINT
DEFINE_ACT_GRAD_DEPX_OP_ARGMAP(Swish, "swish", "beta");  // NOLINT

DEFINE_ACT_GRAD_DEPX_OP_ARGMAP(STanh,
                               "stanh",
                               "scale_a" comma "scale_b");  // NOLINT

DEFINE_ACT_GRAD_DEPOUT_OP_ARGMAP(Relu6, "relu6", "threshold");  // NOLINT

KernelSignature HardSwishOpArgumentMapping(const ArgumentMappingContext& ctx) {
  return KernelSignature(
      "hard_swish_raw", {"X"}, {"threshold", "scale", "offset"}, {"Out"});
}

KernelSignature SwishOpArgumentMapping(const ArgumentMappingContext& ctx) {
  return KernelSignature("swish_raw", {"X"}, {"beta"}, {"Out"});
}

KernelSignature Relu6OpArgumentMapping(const ArgumentMappingContext& ctx) {
  return KernelSignature("relu6_raw", {"X"}, {"threshold"}, {"Out"});
}

KernelSignature PowOpArgumentMapping(const ArgumentMappingContext& ctx) {
  if (ctx.HasInput("FactorTensor")) {
    return KernelSignature("pow", {"X"}, {"FactorTensor"}, {"Out"});
  } else {
    return KernelSignature("pow", {"X"}, {"factor"}, {"Out"});
  }
}

KernelSignature PowGradOpArgumentMapping(const ArgumentMappingContext& ctx) {
  if (ctx.HasInput("FactorTensor")) {
    return KernelSignature(
        "pow_grad", {"X", "Out@GRAD"}, {"FactorTensor"}, {"X@GRAD"});
  } else {
    return KernelSignature(
        "pow_grad", {"X", "Out@GRAD"}, {"factor"}, {"X@GRAD"});
  }
}

KernelSignature PowDoubleGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  if (ctx.HasInput("FactorTensor")) {
    return KernelSignature("pow_double_grad",
                           {"X", "DOut", "DDX"},
                           {"FactorTensor"},
                           {"DX", "DDOut"});
  } else {
    return KernelSignature(
        "pow_double_grad", {"X", "DOut", "DDX"}, {"factor"}, {"DX", "DDOut"});
  }
}

KernelSignature PowTripleGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  if (ctx.HasInput("FactorTensor")) {
    return KernelSignature("pow_triple_grad",
                           {"X", "DOut", "DDX", "D_DX", "D_DDOut"},
                           {"FactorTensor"},
                           {"D_X", "D_DOut", "D_DDX"});
  } else {
    return KernelSignature("pow_triple_grad",
                           {"X", "DOut", "DDX", "D_DX", "D_DDOut"},
                           {"factor"},
                           {"D_X", "D_DOut", "D_DDX"});
  }
}
}  // namespace phi

PD_REGISTER_BASE_KERNEL_NAME(brelu, hard_tanh);
PD_REGISTER_BASE_KERNEL_NAME(brelu_grad, hard_tanh_grad);

PD_REGISTER_ARG_MAPPING_FN(mish_grad, phi::MishGradOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(stanh_grad, phi::STanhGradOpArgumentMapping);

PD_REGISTER_ARG_MAPPING_FN(brelu_grad, phi::HardTanhGradOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(relu6_grad, phi::Relu6GradOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(relu6, phi::Relu6OpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(hard_swish_grad,
                           phi::HardSwishGradOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(hard_swish, phi::HardSwishOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(swish_grad, phi::SwishGradOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(swish, phi::SwishOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(pow_grad, phi::PowGradOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(pow_double_grad,
                           phi::PowDoubleGradOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(pow_triple_grad,
                           phi::PowTripleGradOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(pow, phi::PowOpArgumentMapping);
