// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

KernelSignature SpectralNormOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature("spectral_norm",
                         {"Weight", "U", "V"},
                         {"dim", "power_iters", "eps"},
                         {"Out"});
}

KernelSignature SpectralNormGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature("spectral_norm_grad",
                         {"Weight", "U", "V", "Out@GRAD"},
                         {"dim", "power_iters", "eps"},
                         {"Weight@GRAD"});
}

}  // namespace phi

PD_REGISTER_ARG_MAPPING_FN(spectral_norm, phi::SpectralNormOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(spectral_norm_grad,
                           phi::SpectralNormGradOpArgumentMapping);
