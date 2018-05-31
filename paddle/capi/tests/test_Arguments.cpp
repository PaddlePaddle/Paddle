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

#include <functional>
#include "capi.h"
#include "gtest/gtest.h"
#include "paddle/utils/ThreadLocal.h"

static std::vector<paddle_real> randomBuffer(size_t bufSize) {
  auto& eng = paddle::ThreadLocalRandomEngine::get();
  std::uniform_real_distribution<paddle_real> dist(-1.0, 1.0);
  std::vector<paddle_real> retv;
  retv.reserve(bufSize);
  for (size_t i = 0; i < bufSize; ++i) {
    retv.push_back(dist(eng));
  }
  return retv;
}

TEST(CAPIArguments, create) {
  //! TODO(yuyang18): Test GPU Code.
  paddle_arguments args = paddle_arguments_create_none();
  uint64_t size;
  ASSERT_EQ(kPD_NO_ERROR, paddle_arguments_get_size(args, &size));
  ASSERT_EQ(0UL, size);
  ASSERT_EQ(kPD_NO_ERROR, paddle_arguments_destroy(args));
}

TEST(CAPIArguments, value) {
  paddle_arguments args = paddle_arguments_create_none();
  ASSERT_EQ(kPD_NO_ERROR, paddle_arguments_resize(args, 1));

  paddle_matrix mat = paddle_matrix_create(128, 64, false);
  for (size_t i = 0; i < 128; ++i) {
    std::vector<paddle_real> sampleBuf = randomBuffer(64);
    paddle_matrix_set_row(mat, i, sampleBuf.data());
  }
  ASSERT_EQ(kPD_NO_ERROR, paddle_arguments_set_value(args, 0, mat));

  paddle_matrix val = paddle_matrix_create_none();

  ASSERT_EQ(kPD_NO_ERROR, paddle_arguments_get_value(args, 0, val));

  for (size_t i = 0; i < 128; ++i) {
    paddle_real* row1;
    paddle_real* row2;

    ASSERT_EQ(kPD_NO_ERROR, paddle_matrix_get_row(mat, i, &row1));
    ASSERT_EQ(kPD_NO_ERROR, paddle_matrix_get_row(val, i, &row2));
    ASSERT_EQ(row1, row2);
  }

  paddle_ivector ivec = paddle_ivector_create_none();
  ASSERT_EQ(kPD_NO_ERROR, paddle_ivector_destroy(ivec));
  ASSERT_EQ(kPD_NO_ERROR, paddle_matrix_destroy(val));
  ASSERT_EQ(kPD_NO_ERROR, paddle_matrix_destroy(mat));
  ASSERT_EQ(kPD_NO_ERROR, paddle_arguments_destroy(args));
}

TEST(CAPIArguments, ids) {
  paddle_arguments args = paddle_arguments_create_none();
  ASSERT_EQ(kPD_NO_ERROR, paddle_arguments_resize(args, 1));

  paddle_ivector ivec;
  int array[3] = {1, 2, 3};
  ivec = paddle_ivector_create(array, 3, true, false);
  ASSERT_EQ(kPD_NO_ERROR, paddle_arguments_set_ids(args, 0, ivec));

  paddle_ivector val = paddle_ivector_create_none();
  ASSERT_EQ(kPD_NO_ERROR, paddle_arguments_get_ids(args, 0, val));
  ASSERT_EQ(kPD_NO_ERROR, paddle_ivector_destroy(ivec));
  ASSERT_EQ(kPD_NO_ERROR, paddle_ivector_destroy(val));
  ASSERT_EQ(kPD_NO_ERROR, paddle_arguments_destroy(args));
}

template <typename T1, typename T2>
void testSequenceHelper(T1 setter, T2 getter) {
  paddle_arguments args = paddle_arguments_create_none();
  ASSERT_EQ(kPD_NO_ERROR, paddle_arguments_resize(args, 1));

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
  ASSERT_EQ(kPD_NO_ERROR, paddle_arguments_destroy(args));
}

TEST(CAPIArguments, Sequence) {
  auto testSequence = [](uint32_t nestedLevel) {
    testSequenceHelper(std::bind(paddle_arguments_set_sequence_start_pos,
                                 std::placeholders::_1,
                                 std::placeholders::_2,
                                 nestedLevel,
                                 std::placeholders::_3),
                       std::bind(paddle_arguments_get_sequence_start_pos,
                                 std::placeholders::_1,
                                 std::placeholders::_2,
                                 nestedLevel,
                                 std::placeholders::_3));
  };
  for (uint32_t i = 0; i < 2; ++i) {  // test seq and sub-seq.
    testSequence(i);
  }
}
