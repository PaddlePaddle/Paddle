// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <memory>
#include "paddle/fluid/memory/allocation/allocator.h"

namespace paddle {
namespace memory {
namespace allocation {

class NumpyAllocation : public memory::Allocation {
 public:
  explicit NumpyAllocation(std::shared_ptr<pybind11::array> arr, ssize_t size,
                           paddle::platform::CPUPlace place)
      : Allocation(
            const_cast<void *>(reinterpret_cast<const void *>(arr->data())),
            size, place),
        arr_(arr) {}
  ~NumpyAllocation() override {
    py::gil_scoped_acquire gil;
    arr_->dec_ref();
  }

 private:
  std::shared_ptr<pybind11::array> arr_;
};
}  // namespace allocation
}  // namespace memory
}  // namespace paddle
