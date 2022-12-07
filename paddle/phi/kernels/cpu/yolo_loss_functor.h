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

template <typename T>
struct Box {
  T x, y, w, h;
};

template <typename T>
static inline T sigmoid(T x) {
  return 1.0 / (1.0 + std::exp(-x));
}

template <typename T>
static inline Box<T> GetGtBox(const T* gt, int batch, int max_boxes, int idx) {
  Box<T> b;
  b.x = gt[(batch * max_boxes + idx) * 4];
  b.y = gt[(batch * max_boxes + idx) * 4 + 1];
  b.w = gt[(batch * max_boxes + idx) * 4 + 2];
  b.h = gt[(batch * max_boxes + idx) * 4 + 3];
  return b;
}

static inline int GetEntryIndex(int batch,
                                int an_idx,
                                int hw_idx,
                                int an_num,
                                int an_stride,
                                int stride,
                                int entry) {
  return (batch * an_num + an_idx) * an_stride + entry * stride + hw_idx;
}

}  // namespace phi
