/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/platform/device/mlu/enforce.h"

#include <list>

#include "gtest/gtest.h"

#ifdef PADDLE_WITH_MLU
template <typename T>
bool CheckMluStatusSuccess(T value, const std::string& msg = "success") {
  PADDLE_ENFORCE_MLU_SUCCESS(value);
  return true;
}

template <typename T>
bool CheckMluStatusFailure(T value, const std::string& msg) {
  try {
    PADDLE_ENFORCE_MLU_SUCCESS(value);
    return false;
  } catch (paddle::platform::EnforceNotMet& error) {
    std::string ex_msg = error.what();
    std::cout << ex_msg << std::endl;
    return ex_msg.find(msg) != std::string::npos;
  }
}

TEST(mlu_enforce, mlu_success) {
  EXPECT_TRUE(CheckMluStatusSuccess(cnrtSuccess));
  EXPECT_TRUE(CheckMluStatusFailure(cnrtErrorArgsInvalid, "invalid argument"));
  EXPECT_TRUE(CheckMluStatusFailure(cnrtErrorMemcpyDirectionInvalid,
                                    "invalid memcpy direction"));
  EXPECT_TRUE(
      CheckMluStatusFailure(cnrtErrorDeviceInvalid, "invalid device ordinal"));

  EXPECT_TRUE(CheckMluStatusSuccess(CNNL_STATUS_SUCCESS));
  EXPECT_TRUE(CheckMluStatusFailure(CNNL_STATUS_NOT_INITIALIZED, "CNNL error"));
  EXPECT_TRUE(CheckMluStatusFailure(CNNL_STATUS_ALLOC_FAILED, "CNNL error"));
  EXPECT_TRUE(CheckMluStatusFailure(CNNL_STATUS_BAD_PARAM, "CNNL error"));
  EXPECT_TRUE(CheckMluStatusFailure(CNNL_STATUS_INTERNAL_ERROR, "CNNL error"));

  EXPECT_TRUE(CheckMluStatusSuccess(CN_SUCCESS));
  EXPECT_TRUE(CheckMluStatusFailure(
      CN_ERROR_NOT_READY,
      "Asynchronous operations issued previously not completed yet"));
  EXPECT_TRUE(
      CheckMluStatusFailure(CN_ERROR_NOT_INITIALIZED, "initialization error"));
  EXPECT_TRUE(
      CheckMluStatusFailure(CN_ERROR_INVALID_VALUE, "invalid argument"));
  EXPECT_TRUE(CheckMluStatusFailure(CN_MEMORY_ERROR_OUT_OF_MEMORY,
                                    "device has no memory to alloc"));
}
#endif
