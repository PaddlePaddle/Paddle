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

#pragma once

namespace phi {
namespace funcs {

//////// CalcOutputSize Functor ///////
inline int64_t CalcOutputSize(int64_t input_size,
                              int64_t filter_size,
                              int64_t dilation,
                              int64_t padding1,
                              int64_t padding2,
                              int64_t stride) {
  const int64_t dkernel = dilation * (filter_size - 1) + 1;
  int64_t output_size =
      (input_size + padding1 + padding2 - dkernel) / stride + 1;
  return input_size == -1 ? -1 : output_size;
}

}  // namespace funcs
}  // namespace phi
