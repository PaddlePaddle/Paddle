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

#pragma once

#include "paddle/phi/core/distributed/auto_parallel/inferspmd_utils.h"

#include "paddle/phi/infermeta/spmd_rules/matmul.h"

/**
 * Design Notes:
 *
 * 1. SPMD info is the special meta info of DistTensor, so we put Spmd infer
 * functions in `infermeta` directory.
 *
 * 2. Since the infer functions of Spmd forward and backward are closely related
 * and need to be registered together, we manage them together in one file.
 *
 * 3. SPMD rules are less than infermeta function, and we manage files by
 * operator.
 *
 * 4. The previous registration used some compile-time regular matching methods,
 * which was less flexible, and the registration of SPMD rules here is declare
 * directly in the header file.
 */

namespace phi {
namespace distributed {

// matmul rule
PD_REGISTER_SPMD_RULE(matmul,
                      PD_INFER_SPMD(phi::distributed::MatmulSpmdInferForward),
                      PD_INFER_SPMD(phi::distributed::MatmulSpmdInferBackward));

}  // namespace distributed
}  // namespace phi
