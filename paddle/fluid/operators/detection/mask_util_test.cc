/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/detection/mask_util.h"
#include <gtest/gtest.h>
#include "paddle/fluid/memory/memory.h"

namespace paddle {
namespace operators {

template <typename T>
void Compare(const T* a, const T* b, const int n) {
  for (int i = 0; i < n; i++) {
    EXPECT_EQ(a[i], b[i]);
  }
}

TEST(MaskUtil, Poly2MaskTest) {
  float polys[] = {1.97f, 1.88f, 5.81f, 1.88f, 1.69f,
                   6.53f, 5.94f, 6.38f, 1.97f, 1.88f};
  int h = 8, w = 8;
  int k = 5;  // length(polys) / 2
  // clang-format off
  uint8_t expect_mask[] = {
      0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 1, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 1, 0, 0, 0, 0,
      0, 0, 1, 1, 1, 0, 0, 0,
      0, 0, 1, 1, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0
  };
  // clang-format on

  // the groud-truth mask is computed by coco API:
  //
  // import pycocotools.mask as mask_util
  // import numpy as np
  // segm = [1.97, 1.88, 5.81, 1.88, 1.69, 6.53, 5.94, 6.38, 1.97, 1.88]
  // rles = mask_util.frPyObjects([segm], im_h, im_w)
  // mask = mask_util.decode(rles)
  // print mask
  platform::CPUPlace cpu;
  auto allocation = memory::Alloc(cpu, sizeof(expect_mask));
  uint8_t* mask = reinterpret_cast<uint8_t*>(allocation->ptr());
  Poly2Mask(polys, k, h, w, mask);
  Compare<uint8_t>(expect_mask, mask, h * w);
}

TEST(MaskUtil, Poly2BoxesTest) {
  // clang-format off
  std::vector<std::vector<std::vector<float>>> polys = {
      {{1.97f, 1.88f, 5.81f, 1.88f, 1.69f, 6.53f, 5.94f, 6.38f, 1.97f, 1.88f}},
      {{2.97f, 1.88f, 3.81f, 1.68f, 1.69f, 6.63f, 6.94f, 6.58f, 2.97f, 0.88f}}
  };
  float expect_boxes[] = {
      1.69f, 1.88f, 5.94f, 6.53f,
      1.69f, 0.88f, 6.94f, 6.63f
  };
  // clang-format on

  platform::CPUPlace cpu;
  auto allocation = memory::Alloc(cpu, sizeof(expect_boxes));
  float* boxes = reinterpret_cast<float*>(allocation->ptr());
  Poly2Boxes(polys, boxes);
  Compare<float>(expect_boxes, boxes, 8);
}

TEST(MaskUtil, Polys2MaskWrtBoxTest) {
  // clang-format off
  std::vector<std::vector<std::vector<float>>> polys = {{
      {1.97f, 1.88f, 5.81f, 1.88f, 1.69f, 6.53f, 5.94f, 6.38f, 1.97f, 1.88f},
      {2.97f, 1.88f, 3.81f, 1.68f, 1.69f, 6.63f, 6.94f, 6.58f, 2.97f, 0.88f}}};
  float expect_boxes[] = {
      1.69f, 0.88f, 6.94f, 6.63f
  };
  uint8_t expect_mask[] = {
      0, 0, 0, 0, 0, 0, 0, 0,
      0, 1, 1, 1, 1, 1, 0, 0,
      0, 0, 1, 1, 1, 0, 0, 0,
      0, 0, 1, 1, 1, 0, 0, 0,
      0, 0, 1, 1, 1, 0, 0, 0,
      0, 1, 1, 1, 1, 1, 0, 0,
      0, 1, 1, 1, 1, 1, 1, 0,
      1, 1, 1, 1, 1, 1, 1, 1
  };
  // clang-format on

  platform::CPUPlace cpu;
  auto allocation = memory::Alloc(cpu, sizeof(expect_boxes));
  float* boxes = reinterpret_cast<float*>(allocation->ptr());
  Poly2Boxes(polys, boxes);
  Compare<float>(expect_boxes, boxes, 4);

  auto allocat_mask = memory::Alloc(cpu, sizeof(expect_mask));
  uint8_t* mask = reinterpret_cast<uint8_t*>(allocat_mask->ptr());
  int M = 8;
  Polys2MaskWrtBox(polys[0], expect_boxes, M, mask);
  Compare<uint8_t>(expect_mask, mask, M * M);
}

}  // namespace operators
}  // namespace paddle
