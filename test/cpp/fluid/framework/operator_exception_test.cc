// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <exception>
#include <stdexcept>
#include <string>
#include <utility>

#include "gtest/gtest.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/phi/common/place.h"

namespace paddle {
namespace framework {

class ExceptionThrownOperator : public OperatorBase {
 public:
  using OperatorBase::OperatorBase;

  template <typename T>
  void SetException(T &&obj) {
    exception_ = std::make_exception_ptr(std::forward<T>(obj));
  }

 protected:
  void RunImpl(const Scope &, const phi::Place &) const override {
    if (exception_) {
      std::rethrow_exception(exception_);
    }
  }

 private:
  std::exception_ptr exception_{nullptr};
};

class StubException : public std::exception {
 public:
  const char *what() const noexcept override { return ""; }
};

template <typename T>
bool ExceptionTestMain(T &&obj, bool set_exception) {
  ExceptionThrownOperator op("", {}, {}, {});
  if (set_exception) {
    op.SetException(std::forward<T>(obj));
  }
  Scope scope;
  try {
    op.Run(scope, phi::CPUPlace());
    return false;
  } catch (T &) {
    return true;
  }
}

TEST(test_operator_exception, test_operator_exception) {
  platform::EnforceNotMet enforce_not_met("", __FILE__, __LINE__);
  ASSERT_TRUE(ExceptionTestMain(enforce_not_met, true));
  ASSERT_FALSE(ExceptionTestMain(enforce_not_met, false));

  platform::EOFException eof("", __FILE__, __LINE__);
  ASSERT_TRUE(ExceptionTestMain(eof, true));
  ASSERT_FALSE(ExceptionTestMain(eof, false));

  StubException stub;
  ASSERT_TRUE(ExceptionTestMain(stub, true));
  ASSERT_FALSE(ExceptionTestMain(stub, false));

  ASSERT_TRUE(ExceptionTestMain(1, true));
  ASSERT_FALSE(ExceptionTestMain(1, false));
}

}  // namespace framework
}  // namespace paddle
