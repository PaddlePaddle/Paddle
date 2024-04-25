// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/phi/core/compat/op_utils.h"

namespace phi {

KernelSignature CorrelationOpArgumentMapping(
    const ArgumentMappingContext& ctx UNUSED) {
  return KernelSignature("correlation",
                         {"Input1", "Input2"},
                         {"pad_size",
                          "kernel_size",
                          "max_displacement",
                          "stride1",
                          "stride2",
                          "corr_type_multiply"},
                         {"Output"});
}

KernelSignature CorrelationGradOpArgumentMapping(
    const ArgumentMappingContext& ctx UNUSED) {
  return KernelSignature("correlation_grad",
                         {"Input1", "Input2", "Output@GRAD"},
                         {"pad_size",
                          "kernel_size",
                          "max_displacement",
                          "stride1",
                          "stride2",
                          "corr_type_multiply"},
                         {"Input1@GRAD", "Input2@GRAD"});
}

}  // namespace phi

PD_REGISTER_ARG_MAPPING_FN(correlation, phi::CorrelationOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(correlation_grad,
                           phi::CorrelationGradOpArgumentMapping);
