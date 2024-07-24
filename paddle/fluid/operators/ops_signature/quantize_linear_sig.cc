/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

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

KernelSignature QuantizeLinearOpArgumentMapping(
    const ArgumentMappingContext& ctx UNUSED) {
  bool is_test = ctx.HasAttr("is_test")
                     ? paddle::any_cast<bool>(ctx.Attr("is_test"))
                     : true;
  if (is_test) {
    return KernelSignature("quantize_linear_deprecated_infer",
                           {"X", "Scale", "ZeroPoint"},
                           {"quant_axis",
                            "bit_length",
                            "qmin",
                            "qmax",
                            "round_type",
                            "only_observer"},
                           {"Y"});
  } else {
    return KernelSignature("quantize_linear_deprecated_train",
                           {"X", "Scale", "ZeroPoint", "InAccum", "InState"},
                           {"quant_axis",
                            "bit_length",
                            "qmin",
                            "qmax",
                            "round_type",
                            "only_observer"},
                           {"Y", "OutState", "OutAccum", "OutScale"});
  }
}

KernelSignature DeQuantizeLinearOpArgumentMapping(
    const ArgumentMappingContext& ctx UNUSED) {
  return KernelSignature("dequantize_linear_deprecated",
                         {"X", "Scale", "ZeroPoint"},
                         {"quant_axis",
                          "bit_length",
                          "qmin",
                          "qmax",
                          "round_type",
                          "only_observer"},
                         {"Y"});
}

}  // namespace phi

PD_REGISTER_ARG_MAPPING_FN(quantize_linear,
                           phi::QuantizeLinearOpArgumentMapping);

PD_REGISTER_ARG_MAPPING_FN(dequantize_linear,
                           phi::DeQuantizeLinearOpArgumentMapping);
