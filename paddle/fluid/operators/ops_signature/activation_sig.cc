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
      const ArgumentMappingContext& ctx UNUSED) {                 \
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

DEFINE_ACT_GRAD_DEPX_OP_ARGMAP(HardTanh, "hardtanh", "t_min" comma "t_max");
DEFINE_ACT_GRAD_DEPX_OP_ARGMAP(Mish, "mish", "threshold");

KernelSignature HardSwishOpArgumentMapping(
    const ArgumentMappingContext& ctx UNUSED) {
  return KernelSignature("hardswish", {"X"}, {}, {"Out"});
}

}  // namespace phi

PD_REGISTER_BASE_KERNEL_NAME(hard_swish, hardswish);
PD_REGISTER_ARG_MAPPING_FN(mish_grad, phi::MishGradOpArgumentMapping);

PD_REGISTER_ARG_MAPPING_FN(hard_swish, phi::HardSwishOpArgumentMapping);
