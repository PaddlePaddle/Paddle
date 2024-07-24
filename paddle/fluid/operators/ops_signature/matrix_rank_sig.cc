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

// we have to return every specific KernelSignature for inference now
KernelSignature MatrixRankOpArgumentMapping(const ArgumentMappingContext& ctx) {
  if (ctx.IsForInferShape()) {
    return KernelSignature("matrix_rank_tol",
                           {"X", "TolTensor"},
                           {"use_default_tol", "hermitian"},
                           {"Out"});
  }
  if (ctx.HasInput("TolTensor")) {
    return KernelSignature("matrix_rank_tol",
                           {"X", "TolTensor"},
                           {"use_default_tol", "hermitian"},
                           {"Out"});
  } else {
    return KernelSignature("matrix_rank",
                           {"X"},
                           {
                               "tol",
                               "use_default_tol",
                               "hermitian",
                           },
                           {"Out"});
  }
}

}  // namespace phi

PD_REGISTER_ARG_MAPPING_FN(matrix_rank, phi::MatrixRankOpArgumentMapping);
