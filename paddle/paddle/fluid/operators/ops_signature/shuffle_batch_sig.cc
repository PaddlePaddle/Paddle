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

KernelSignature ShuffleBatchOpArgumentMapping(
    const ArgumentMappingContext& ctx UNUSED) {
  return KernelSignature("shuffle_batch",
                         {"X", "Seed"},
                         {"startup_seed"},
                         {"Out", "ShuffleIdx", "SeedOut"});
}

KernelSignature ShuffleBatchGradOpArgumentMapping(
    const ArgumentMappingContext& ctx UNUSED) {
  return KernelSignature("shuffle_batch_grad",
                         {"ShuffleIdx", "Out@GRAD"},
                         {"startup_seed"},
                         {"X@GRAD"});
}

}  // namespace phi

PD_REGISTER_ARG_MAPPING_FN(shuffle_batch, phi::ShuffleBatchOpArgumentMapping);

PD_REGISTER_ARG_MAPPING_FN(shuffle_batch_grad,
                           phi::ShuffleBatchGradOpArgumentMapping);
