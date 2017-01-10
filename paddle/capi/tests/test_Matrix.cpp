/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "PaddleCAPI.h"
#include "gtest/gtest.h"

TEST(CAPIMatrix, create) {
  PD_Matrix mat;
  ASSERT_EQ(kPD_NO_ERROR, PDMatCreate(&mat, 128, 32, false));
  std::vector<pd_real> sampleRow;
  sampleRow.resize(32);
  for (size_t i = 0; i < sampleRow.size(); ++i) {
    sampleRow[i] = 1.0 / (i + 1.0);
  }
  ASSERT_EQ(kPD_NO_ERROR, PDMatCopyToRow(mat, 0, sampleRow.data()));
  ASSERT_EQ(kPD_OUT_OF_RANGE, PDMatCopyToRow(mat, 128, sampleRow.data()));

  pd_real* arrayPtr;

  ASSERT_EQ(kPD_NO_ERROR, PDMatGetRow(mat, 0, &arrayPtr));
  for (size_t i = 0; i < sampleRow.size(); ++i) {
    ASSERT_NEAR(sampleRow[i], arrayPtr[i], 1e-5);
  }

  uint64_t height, width;
  ASSERT_EQ(kPD_NO_ERROR, PDMatGetShape(mat, &height, &width));
  ASSERT_EQ(128UL, height);
  ASSERT_EQ(32UL, width);
  ASSERT_EQ(kPD_NO_ERROR, PDMatDestroy(mat));
}

TEST(CAPIMatrix, createNone) {
  PD_Matrix mat;
  ASSERT_EQ(kPD_NO_ERROR, PDMatCreateNone(&mat));
  ASSERT_EQ(kPD_NO_ERROR, PDMatDestroy(mat));
}
