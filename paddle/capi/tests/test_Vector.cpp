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

TEST(CAPIVector, create) {
  PD_IVector vec;
  int array[3] = {1, 2, 3};
  ASSERT_EQ(kPD_NO_ERROR, PDIVectorCreate(&vec, array, 3, true));
  ASSERT_EQ(kPD_NO_ERROR, PDIVectorCreate(&vec, array, 3, false));
  ASSERT_EQ(kPD_NO_ERROR, PDIVectorResize(vec, 1000));
  uint64_t size;
  ASSERT_EQ(kPD_NO_ERROR, PDIVectorGetSize(vec, &size));
  ASSERT_EQ(kPD_NO_ERROR, PDIVecDestroy(vec));
}

TEST(CAPIVector, createNone) {
  PD_IVector vec;
  ASSERT_EQ(kPD_NO_ERROR, PDIVecCreateNone(&vec));
  ASSERT_EQ(kPD_NO_ERROR, PDIVecDestroy(vec));
}
