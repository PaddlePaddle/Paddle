// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/backends/gpu/gpu_context.h"

namespace phi {
namespace funcs {

// This functor involves a fusion calculation in Ernie or Bert.
//  The fusion mode is as follows:
//
//      in_var  emb       in_var   emb
//        |      |          |       |
//      lookup_table      lookup_table
//            |                 |
//         lkt_var           lkt_var
//             \                /
//              elementwise_add
//                     |
//                elt_out_var
//
template <typename T>
class EmbEltwiseLayerNormFunctor {
 public:
  void operator()(int batch,
                  int seq_len,
                  int hidden,
                  const int64_t* ids,
                  const T* scale,
                  const T* bias,
                  const int64_t* embs,
                  T* output,
                  float eps,
                  int input_num,
                  gpuStream_t stream);
};
}  // namespace funcs
}  // namespace phi
