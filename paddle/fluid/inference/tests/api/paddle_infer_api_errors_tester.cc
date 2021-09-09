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

#include "gflags/gflags.h"
#include "glog/logging.h"
#include "gtest/gtest.h"

#include "paddle/fluid/inference/api/paddle_infer_contrib.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle_infer {
namespace contrib {

TEST(Status, ctor) { CHECK(Status::OK().ok()); }

struct FakeException {
  void pd_exception(int a, const std::string& msg) const {
    PADDLE_ENFORCE_NE(a, a, paddle::platform::errors::InvalidArgument(msg));
  }
  void base_exception() const { throw std::exception(); }
};

TEST(Status, pd_exception) {
  FakeException e;
  Status status = status_wrapper([&]() { e.pd_exception(1, "test"); });
  CHECK(!status.ok());
  CHECK_NE(status.code(), 0);
}

TEST(Status, basic_exception) {
  FakeException e;
  Status status;
  status = status_wrapper([&]() { e.base_exception(); });
  CHECK(!status.ok());
}

}  // namespace contrib
}  // namespace paddle_infer
