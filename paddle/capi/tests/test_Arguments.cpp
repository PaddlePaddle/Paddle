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

  PD_Matrix mat = PDMatCreate(128, 64, false);
  for (size_t i = 0; i < 128; ++i) {
    std::vector<pd_real> sampleBuf = randomBuffer(64);
    PDMatCopyToRow(mat, i, sampleBuf.data());
  }
  ASSERT_EQ(kPD_NO_ERROR, PDArgsSetValue(args, 0, mat));

  PD_Matrix val = PDMatCreateNone();

  ASSERT_EQ(kPD_NO_ERROR, PDArgsGetValue(args, 0, val));

  for (size_t i = 0; i < 128; ++i) {
    pd_real* row1;
    pd_real* row2;

    ASSERT_EQ(kPD_NO_ERROR, PDMatGetRow(mat, i, &row1));
    ASSERT_EQ(kPD_NO_ERROR, PDMatGetRow(val, i, &row2));
    ASSERT_EQ(row1, row2);
  }

  PD_IVector ivec = PDIVecCreateNone();
  ASSERT_EQ(kPD_NO_ERROR, PDIVecDestroy(ivec));
  ASSERT_EQ(kPD_NO_ERROR, PDMatDestroy(val));
  ASSERT_EQ(kPD_NO_ERROR, PDMatDestroy(mat));
  ASSERT_EQ(kPD_NO_ERROR, PDArgsDestroy(args));
}

TEST(CAPIArguments, ids) {
  PD_Arguments args = PDArgsCreateNone();
  ASSERT_EQ(kPD_NO_ERROR, PDArgsResize(args, 1));

  PD_IVector ivec;
  int array[3] = {1, 2, 3};
  ivec = PDIVectorCreate(array, 3, true, false);
  ASSERT_EQ(kPD_NO_ERROR, PDArgsSetIds(args, 0, ivec));

  PD_IVector val = PDIVecCreateNone();
  ASSERT_EQ(kPD_NO_ERROR, PDArgsGetIds(args, 0, val));
  ASSERT_EQ(kPD_NO_ERROR, PDIVecDestroy(ivec));
  ASSERT_EQ(kPD_NO_ERROR, PDIVecDestroy(val));
  ASSERT_EQ(kPD_NO_ERROR, PDArgsDestroy(args));
}

template <typename T1, typename T2>
void testSequenceHelper(T1 setter, T2 getter) {
  PD_Arguments args = PDArgsCreateNone();
  ASSERT_EQ(kPD_NO_ERROR, PDArgsResize(args, 1));

  PD_IVector ivec;
  int array[3] = {1, 2, 3};
  ivec = PDIVectorCreate(array, 3, true, false);
  ASSERT_EQ(kPD_NO_ERROR, setter(args, 0, ivec));

  PD_IVector val = PDIVecCreateNone();
  ASSERT_EQ(kPD_NO_ERROR, getter(args, 0, val));
  uint64_t size;
  ASSERT_EQ(kPD_NO_ERROR, PDIVectorGetSize(val, &size));

  int* rawBuf;
  ASSERT_EQ(kPD_NO_ERROR, PDIVectorGet(val, &rawBuf));
  for (size_t i = 0; i < size; ++i) {
    ASSERT_EQ(array[i], rawBuf[i]);
  }

  ASSERT_EQ(kPD_NO_ERROR, PDIVecDestroy(ivec));
  ASSERT_EQ(kPD_NO_ERROR, PDIVecDestroy(val));
  ASSERT_EQ(kPD_NO_ERROR, PDArgsDestroy(args));
}

TEST(CAPIArguments, Sequence) {
  testSequenceHelper(PDArgsSetSequenceStartPos, PDArgsGetSequenceStartPos);
  testSequenceHelper(PDArgsSetSubSequenceStartPos,
                     PDArgsGetSubSequenceStartPos);
}
