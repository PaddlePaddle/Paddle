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

#include "capi.h"
#include "gtest/gtest.h"

TEST(CAPIMatrix, create) {
  //! TODO(yuyang18): Test GPU Code.
  paddle_matrix mat = paddle_matrix_create(128, 32, false);
  std::vector<paddle_real> sampleRow;
  sampleRow.resize(32);
  for (size_t i = 0; i < sampleRow.size(); ++i) {
    sampleRow[i] = 1.0 / (i + 1.0);
  }
  ASSERT_EQ(kPD_NO_ERROR, paddle_matrix_set_row(mat, 0, sampleRow.data()));
  ASSERT_EQ(kPD_OUT_OF_RANGE,
            paddle_matrix_set_row(mat, 128, sampleRow.data()));

  paddle_real* arrayPtr;

  ASSERT_EQ(kPD_NO_ERROR, paddle_matrix_get_row(mat, 0, &arrayPtr));
  for (size_t i = 0; i < sampleRow.size(); ++i) {
    ASSERT_NEAR(sampleRow[i], arrayPtr[i], 1e-5);
  }

  uint64_t height, width;
  ASSERT_EQ(kPD_NO_ERROR, paddle_matrix_get_shape(mat, &height, &width));
  ASSERT_EQ(128UL, height);
  ASSERT_EQ(32UL, width);
  ASSERT_EQ(kPD_NO_ERROR, paddle_matrix_destroy(mat));
}

TEST(CAPIMatrix, createNone) {
  paddle_matrix mat = paddle_matrix_create_none();
  ASSERT_EQ(kPD_NO_ERROR, paddle_matrix_destroy(mat));
}

TEST(CAPIMatrix, cpu_get_set_value) {
  paddle_matrix mat = paddle_matrix_create(128, 32, false);
  std::vector<paddle_real> sample;
  std::vector<paddle_real> result;
  sample.resize(128 * 32);
  result.resize(128 * 32);
  for (size_t i = 0; i < sample.size(); ++i) {
    sample[i] = 1.0 / (i + 1.0);
  }
  ASSERT_EQ(kPD_NO_ERROR, paddle_matrix_set_value(mat, sample.data()));
  ASSERT_EQ(kPD_NO_ERROR, paddle_matrix_get_value(mat, result.data()));
  for (size_t i = 0; i < sample.size(); ++i) {
    ASSERT_NEAR(sample[i], result[i], 1e-5);
  }

  uint64_t height, width;
  ASSERT_EQ(kPD_NO_ERROR, paddle_matrix_get_shape(mat, &height, &width));
  ASSERT_EQ(128UL, height);
  ASSERT_EQ(32UL, width);
  ASSERT_EQ(kPD_NO_ERROR, paddle_matrix_destroy(mat));
}

#ifdef PADDLE_WITH_CUDA
TEST(CAPIMatrix, gpu_get_set_value) {
  paddle_matrix mat = paddle_matrix_create(128, 32, true);
  std::vector<paddle_real> sample;
  std::vector<paddle_real> result;
  sample.resize(128 * 32);
  result.resize(128 * 32);
  for (size_t i = 0; i < sample.size(); ++i) {
    sample[i] = 1.0 / (i + 1.0);
  }
  ASSERT_EQ(kPD_NO_ERROR, paddle_matrix_set_value(mat, sample.data()));
  ASSERT_EQ(kPD_NO_ERROR, paddle_matrix_get_value(mat, result.data()));
  for (size_t i = 0; i < sample.size(); ++i) {
    ASSERT_NEAR(sample[i], result[i], 1e-5);
  }

  uint64_t height, width;
  ASSERT_EQ(kPD_NO_ERROR, paddle_matrix_get_shape(mat, &height, &width));
  ASSERT_EQ(128UL, height);
  ASSERT_EQ(32UL, width);
  ASSERT_EQ(kPD_NO_ERROR, paddle_matrix_destroy(mat));
}
#endif
