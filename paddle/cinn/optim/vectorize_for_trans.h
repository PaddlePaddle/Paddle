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

#pragma once
#include "paddle/cinn/ir/ir.h"
namespace cinn {
namespace optim {
/**
 * Deal with forOp with vectorization.
 * if vectorize factor match vectorize instruction and don't have adjaccnt
 * if-block.
 *
 * e.g.
 *
 * serial for (i, 0, 4)
 *  serial for (j, 0, 4)
 *    vectorize[4] for (v1, 0, 4)
 *      float a[i, j, v1] = float b[i, j, v1] + float c[i, j, v1]
 *
 * to
 *
 * serial for (i, 0, 4)
 *  serial for (j, 0, 4)
 *    float4* temp_0_ptr = float4<4>*(get_addr(a[i * 4 + j]))
 *    float4 temp_1
 *    float4 temp_2 = b[i * 4 + j]
 *    float4 temp_3 = c[i * 4 + j]
 *    temp_1[0] = temp_2[0] + temp_3[0]
 *    temp_1[1] = temp_2[1] + temp_3[1]
 *    temp_1[2] = temp_2[2] + temp_3[2]
 *    temp_1[3] = temp_2[3] + temp_3[3]
 *    temp_0_ptr[0] = temp_1
 */
void VectorizeForTrans(Expr *expr);
}  // namespace optim
}  // namespace cinn
