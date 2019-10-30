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

#pragma once

#include <pybind11/numpy.h>
#include <memory>
#include "paddle/fluid/memory/allocation/allocator.h"
#include "pybind11/pybind11.h"

namespace py = pybind11;

namespace paddle {
namespace memory {
namespace allocation {

class NumpyAllocation : public Allocation {
 public:
  NumpyAllocation(const py::array *arr, const ssize_t size)
      : arr_(arr),
        Allocation(
            const_cast<void *>(reinterpret_cast<const void *>(arr->data())),
            size, platform::CPUPlace()) {}
  ~NumpyAllocation() {
    py::gil_scoped_acquire gil;
    arr_->dec_ref();
  }

 private:
  const py::array *arr_;
};

std::shared_ptr<Allocation> FromNumpyArray(const py::array *arr, ssize_t size) {
  return std::make_shared<NumpyAllocation>(arr, size);
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
