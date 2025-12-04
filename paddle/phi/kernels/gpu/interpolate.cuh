// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

template <typename MT>
__device__ __forceinline__ void ComputeWeightsSpan(const int i,
                                                   const int input_size,
                                                   const MT scale,
                                                   const MT support,
                                                   int* xmin,
                                                   int* xsize,
                                                   MT* center) {
  *center = scale * (i + static_cast<MT>(0.5));
  *xmin = max(static_cast<int>(*center - support + static_cast<MT>(0.5)), 0);
  *xsize = min(static_cast<int>(*center + support + static_cast<MT>(0.5)),
               input_size) -
           *xmin;
}

}  // namespace phi
