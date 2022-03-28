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

/**
 * Note [ Why does the ArgumentMapping function need to be so complicated? ]
 *
 * In order to meet the requirements of infrt, the function used to match Op
 * and Kernel parameters, need to be placed in phi as a compatible component,
 * and does not depend on fluid.
 *
 * Because infrt not only needs to dynamically call this argument mapping
 * function at runtime, but also needs to statically declare all possible
 * results of the function before running without any information.
 *
 * The infrt declare like:
 *
 * def PDKEL_Reshape_to_CPU : Pat<
 *     (PD_ReshapeOp $x, $shape_tensor， $shape_attr), // OpMaker arguements
 *     (PDKEL_ReshapeKernelAttr $x, fn($shape_attr)>;  // Kernel arguments
 * def PDKEL_Reshape_to_CPU : Pat<
 *     (PD_ReshapeOp $x, $shape_tensor， $shape_attr),
 *     (PDKEL_ReshapeKernelAttr $x, fn($shape_tensor)>;
 *
 * Therefore, we need to write out each result of the argument mapping function,
 * like `KernelSignature("full", {}, {"ShapeTensor", "value"}, {"Out"})`, it
 * cannot contains variable, only can contains const char* string.
 *
 * Infrt will parse all results before running for the generation of the above
 * static declare, which leads to some functions being written in a long way,
 * and the complicated ones may have hundreds of lines, which has certain side
 * effects on the programming experience.
 */
KernelSignature ScaleOpArgumentMapping(const ArgumentMappingContext& ctx) {
  if (ctx.IsDenseTensorInput("X")) {
    if (ctx.HasInput("ScaleTensor")) {
      return KernelSignature(
          "scale", {"X"}, {"ScaleTensor", "bias", "bias_after_scale"}, {"Out"});
    } else {
      return KernelSignature(
          "scale", {"X"}, {"scale", "bias", "bias_after_scale"}, {"Out"});
    }
  } else if (ctx.IsSelectedRowsInput("X")) {
    if (ctx.HasInput("ScaleTensor")) {
      return KernelSignature("scale_sr",
                             {"X"},
                             {"ScaleTensor", "bias", "bias_after_scale"},
                             {"Out"});
    } else {
      return KernelSignature(
          "scale_sr", {"X"}, {"scale", "bias", "bias_after_scale"}, {"Out"});
    }
  } else {
    return KernelSignature("unregistered", {}, {}, {});
  }
}

}  // namespace phi

// op_type, api_name, arg_mapping_fn
PD_REGISTER_ARG_MAPPING_FN(scale, phi::ScaleOpArgumentMapping);
