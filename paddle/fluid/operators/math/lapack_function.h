//   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

namespace paddle {
namespace operators {
namespace math {

// LU (for example)
template <typename T>
void lapackLu(int m, int n, T *a, int lda, int *ipiv, int *info);

template <typename T1, typename T2 = T1>
void lapackEig(char jobvl, char jobvr, int n, T1 *a, int lda, T1 *w, T1 *vl,
               int ldvl, T1 *vr, int ldvr, T1 *work, int lwork, T2 *rwork,
               int *info);

}  // namespace math
}  // namespace operators
}  // namespace paddle
