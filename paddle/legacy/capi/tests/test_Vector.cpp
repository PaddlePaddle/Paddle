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

TEST(CAPIVector, create) {
  //! TODO(yuyang18): Test GPU Code.
  paddle_ivector vec;
  int array[3] = {1, 2, 3};
  vec = paddle_ivector_create(array, 3, true, false);
  ASSERT_EQ(kPD_NO_ERROR, paddle_ivector_resize(vec, 1000));
  uint64_t size;
  ASSERT_EQ(kPD_NO_ERROR, paddle_ivector_get_size(vec, &size));
  ASSERT_EQ(kPD_NO_ERROR, paddle_ivector_destroy(vec));
}

TEST(CAPIVector, createNone) {
  paddle_ivector vec = paddle_ivector_create_none();
  ASSERT_EQ(kPD_NO_ERROR, paddle_ivector_destroy(vec));
}
