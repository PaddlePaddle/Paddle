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

KernelSignature SaveCombineOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  if (ctx.IsDenseTensorInputs("X")) {
    return KernelSignature(
        "save_combine_tensor",
        {"X"},
        {"file_path", "overwrite", "save_as_fp16", "save_to_memory"},
        {"Y"});
  } else {
    return KernelSignature(
        "save_combine_vocab",
        {"X"},
        {"file_path", "overwrite", "save_as_fp16", "save_to_memory"},
        {"Y"});
  }
}

}  // namespace phi

PD_REGISTER_ARG_MAPPING_FN(save_combine, phi::SaveCombineOpArgumentMapping);
