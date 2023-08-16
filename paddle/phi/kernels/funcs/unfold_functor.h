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
inline int CalcOutputSize(int input_size,
                          int filter_size,
                          int dilation,
                          int padding1,
                          int padding2,
                          int stride) {
  const int dkernel = dilation * (filter_size - 1) + 1;
  int output_size = (input_size + padding1 + padding2 - dkernel) / stride + 1;
  return input_size == -1 ? -1 : output_size;
}

}  // namespace funcs
}  // namespace phi
