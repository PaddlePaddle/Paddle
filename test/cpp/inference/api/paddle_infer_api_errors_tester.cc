// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "glog/logging.h"
#include "gtest/gtest.h"
#include "paddle/common/flags.h"
#include "paddle/fluid/inference/api/paddle_infer_contrib.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle_infer {
namespace contrib {

TEST(Status, ctor) { CHECK(Status::OK().ok()); }

struct FakeException {
  void pd_exception(int a) const {
    PADDLE_ENFORCE_NE(a,
                      a,
                      phi::errors::InvalidArgument(
                          "This is a preset error message used to verify "
                          "whether the exception meets expectations: %d, %d.",
                          a,
                          a));
  }
  [[noreturn]] void base_exception() const { throw std::exception(); }
  void no_exception() const noexcept {}
};

TEST(Status, pd_exception) {
  FakeException e;
  Status status = get_status([&]() { e.pd_exception(1); });
  CHECK(!status.ok());
  CHECK(status == status);
  CHECK(!(status != status));
  CHECK_EQ(status.code(), phi::ErrorCode::INVALID_ARGUMENT + 1);
  LOG(INFO) << status.error_message();
}

TEST(Status, basic_exception) {
  FakeException e;
  Status status;
  status = get_status([&]() { e.base_exception(); });
  CHECK(!status.ok());
  LOG(INFO) << status.error_message();
}

TEST(Status, no_exception) {
  FakeException e;
  Status status;
  status = get_status([&]() { e.no_exception(); });
  CHECK(status.ok());
}

TEST(Status, copy) {
  Status status;
  Status status_1(status);
  status_1 = status;
}

}  // namespace contrib
}  // namespace paddle_infer
