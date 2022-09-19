// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <pybind11/pybind11.h>

#include <cassert>
#include <future>
#include <string>

#include "paddle/fluid/platform/macros.h"

namespace py = pybind11;
namespace paddle {
namespace distributed {
class FutureWrapper {
 public:
  FutureWrapper() {}
  explicit FutureWrapper(std::future<std::string> fut) : fut_(std::move(fut)) {}
  py::bytes wait() {
    // GIL must be released, otherwise fut_.get() blocking will cause the
    // service to fail to process RPC requests, leading to deadlock
    assert(!PyGILState_Check());
    auto s = fut_.get();
    py::gil_scoped_acquire ag;
    return py::bytes(s);
  }

 private:
  DISABLE_COPY_AND_ASSIGN(FutureWrapper);
  std::future<std::string> fut_;
};
}  // namespace distributed
}  // namespace paddle
