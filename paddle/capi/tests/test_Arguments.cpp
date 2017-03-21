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
#include "paddle/utils/ThreadLocal.h"

static std::vector<pd_real> randomBuffer(size_t bufSize) {
  auto& eng = paddle::ThreadLocalRandomEngine::get();
  std::uniform_real_distribution<pd_real> dist(-1.0, 1.0);
  std::vector<pd_real> retv;
  retv.reserve(bufSize);
  for (size_t i = 0; i < bufSize; ++i) {
    retv.push_back(dist(eng));
  }
  return retv;
}

TEST(CAPIArguments, create) {
  PD_Arguments args = PDArgsCreateNone();
  uint64_t size;
  ASSERT_EQ(kPD_NO_ERROR, PDArgsGetSize(args, &size));
  ASSERT_EQ(0UL, size);
  ASSERT_EQ(kPD_NO_ERROR, PDArgsDestroy(args));
}

TEST(CAPIArguments, value) {
  PD_Arguments args = PDArgsCreateNone();
  ASSERT_EQ(kPD_NO_ERROR, PDArgsResize(args, 1));

  paddle_matrix mat = paddle_matrix_create(128, 64, false);
  for (size_t i = 0; i < 128; ++i) {
    std::vector<pd_real> sampleBuf = randomBuffer(64);
    paddle_matrix_set_row(mat, i, sampleBuf.data());
  }
  ASSERT_EQ(kPD_NO_ERROR, PDArgsSetValue(args, 0, mat));

  paddle_matrix val = paddle_matrix_create_none();

  ASSERT_EQ(kPD_NO_ERROR, PDArgsGetValue(args, 0, val));

  for (size_t i = 0; i < 128; ++i) {
    pd_real* row1;
    pd_real* row2;

    ASSERT_EQ(kPD_NO_ERROR, paddle_matrix_get_row(mat, i, &row1));
    ASSERT_EQ(kPD_NO_ERROR, paddle_matrix_get_row(val, i, &row2));
    ASSERT_EQ(row1, row2);
  }

  paddle_ivector ivec = paddle_ivector_create_none();
  ASSERT_EQ(kPD_NO_ERROR, paddle_ivector_destroy(ivec));
  ASSERT_EQ(kPD_NO_ERROR, paddle_matrix_destroy(val));
  ASSERT_EQ(kPD_NO_ERROR, paddle_matrix_destroy(mat));
  ASSERT_EQ(kPD_NO_ERROR, PDArgsDestroy(args));
}

TEST(CAPIArguments, ids) {
  PD_Arguments args = PDArgsCreateNone();
  ASSERT_EQ(kPD_NO_ERROR, PDArgsResize(args, 1));

  paddle_ivector ivec;
  int array[3] = {1, 2, 3};
  ivec = paddle_ivector_create(array, 3, true, false);
  ASSERT_EQ(kPD_NO_ERROR, PDArgsSetIds(args, 0, ivec));

  paddle_ivector val = paddle_ivector_create_none();
  ASSERT_EQ(kPD_NO_ERROR, PDArgsGetIds(args, 0, val));
  ASSERT_EQ(kPD_NO_ERROR, paddle_ivector_destroy(ivec));
  ASSERT_EQ(kPD_NO_ERROR, paddle_ivector_destroy(val));
  ASSERT_EQ(kPD_NO_ERROR, PDArgsDestroy(args));
}

template <typename T1, typename T2>
void testSequenceHelper(T1 setter, T2 getter) {
  PD_Arguments args = PDArgsCreateNone();
  ASSERT_EQ(kPD_NO_ERROR, PDArgsResize(args, 1));

  paddle_ivector ivec;
  int array[3] = {1, 2, 3};
  ivec = paddle_ivector_create(array, 3, true, false);
  ASSERT_EQ(kPD_NO_ERROR, setter(args, 0, ivec));

  paddle_ivector val = paddle_ivector_create_none();
  ASSERT_EQ(kPD_NO_ERROR, getter(args, 0, val));
  uint64_t size;
  ASSERT_EQ(kPD_NO_ERROR, paddle_ivector_get_size(val, &size));

  int* rawBuf;
  ASSERT_EQ(kPD_NO_ERROR, paddle_ivector_get(val, &rawBuf));
  for (size_t i = 0; i < size; ++i) {
    ASSERT_EQ(array[i], rawBuf[i]);
  }

  ASSERT_EQ(kPD_NO_ERROR, paddle_ivector_destroy(ivec));
  ASSERT_EQ(kPD_NO_ERROR, paddle_ivector_destroy(val));
  ASSERT_EQ(kPD_NO_ERROR, PDArgsDestroy(args));
}

TEST(CAPIArguments, Sequence) {
  testSequenceHelper(PDArgsSetSequenceStartPos, PDArgsGetSequenceStartPos);
  testSequenceHelper(PDArgsSetSubSequenceStartPos,
                     PDArgsGetSubSequenceStartPos);
}
