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

#define DefineActGradDepXOpArgMap(func_name, op_name)                        \
  KernelSignature func_name##GradOpArgumentMapping(                          \
      const ArgumentMappingContext& ctx) {                                   \
    return KernelSignature(                                                  \
        op_name "_grad", {"X", GradVarName("Out")}, {}, {GradVarName("X")}); \
  }

#define DefineActGradDepOutOpArgMap(func_name, op_name)                        \
  KernelSignature func_name##GradOpArgumentMapping(                            \
      const ArgumentMappingContext& ctx) {                                     \
    return KernelSignature(                                                    \
        op_name "_grad", {"Out", GradVarName("Out")}, {}, {GradVarName("X")}); \
  }

KernelSignature ReluDoubleGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature("relu_double_grad", {"Out", "DDX"}, {}, {"DDOut"});
}

DefineActGradDepXOpArgMap(Cos, "cos") 
DefineActGradDepXOpArgMap(Tan, "tan")
DefineActGradDepXOpArgMap(Acos, "acos")
DefineActGradDepXOpArgMap(Sin, "sin") 
DefineActGradDepXOpArgMap(Asin, "asin")
DefineActGradDepXOpArgMap(Atan, "atan")
DefineActGradDepXOpArgMap(Sinh, "sinh")
DefineActGradDepXOpArgMap(Cosh, "cosh")
DefineActGradDepXOpArgMap(Asinh, "asinh")
DefineActGradDepXOpArgMap(Acosh, "acosh")
DefineActGradDepXOpArgMap(Atanh, "atanh")
DefineActGradDepOutOpArgMap(Relu, "relu")
}  // namespace phi

PD_REGISTER_BASE_KERNEL_NAME(relu_grad_grad, relu_double_grad);

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
