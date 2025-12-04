//   Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#ifdef PADDLE_WITH_MAGMA
#pragma once
#include "paddle/phi/backends/dynload/magma.h"

namespace phi {
namespace funcs {

void magmaEnsureInit();

// Eig
template <typename T1, typename T2 = T1>
void magmaEig(magma_vec_t jobvl,
              magma_vec_t jobvr,
              magma_int_t n,
              T1 *a,
              magma_int_t lda,
              T1 *w,
              T1 *vl,
              magma_int_t ldvl,
              T1 *vr,
              magma_int_t ldvr,
              T1 *work,
              magma_int_t lwork,
              T2 *rwork,
              magma_int_t *info);

}  // namespace funcs
}  // namespace phi
#endif
