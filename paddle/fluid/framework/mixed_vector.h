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

#pragma once

#include <algorithm>
#include <initializer_list>
#include <memory>
#include <mutex>  // NOLINT
#include <utility>
#include <vector>

#include "glog/logging.h"
#include "paddle/fluid/framework/details/cow_ptr.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/utils/none.h"
#include "paddle/utils/optional.h"

namespace paddle {
namespace framework {

template <typename T>
class CPUVector : public std::vector<T, std::allocator<T>> {
 public:
  CPUVector() : std::vector<T>() {}
  CPUVector(size_t count, const T &value = T())  // NOLINT
      : std::vector<T>(count, value) {}
  CPUVector(std::initializer_list<T> init) : std::vector<T>(init) {}
  CPUVector(const std::vector<T> &other) : std::vector<T>(other) {}  // NOLINT
  CPUVector(const CPUVector<T> &other) : std::vector<T>(other) {}
  CPUVector(CPUVector<T> &&other) : std::vector<T>(std::move(other)) {}
  CPUVector(std::vector<T> &&other)  // NOLINT
      : std::vector<T>(std::move(other)) {}
  CPUVector &operator=(const CPUVector &other) {
    this->assign(other.begin(), other.end());
    return *this;
  }
  CPUVector &operator=(const std::vector<T> &other) {
    this->assign(other.begin(), other.end());
    return *this;
  }

  friend std::ostream &operator<<(std::ostream &os, const CPUVector<T> &other) {
    std::stringstream ss;
    for (auto v : other) {
      os << v << " ";
    }
    return os;
  }

  T &operator[](size_t id) { return this->at(id); }

  const T &operator[](size_t id) const { return this->at(id); }

  template <typename D>
  void Extend(const D &begin, const D &end) {
    this->reserve(this->size() + size_t(end - begin));
    this->insert(this->end(), begin, end);
  }

  const T *CUDAData(platform::Place place) const {
    PADDLE_THROW(platform::errors::Unavailable(
        "Vector::CUDAData() method is not supported in CPU-only version."));
  }

  T *CUDAMutableData(platform::Place place) {
    PADDLE_THROW(platform::errors::Unavailable(
        "Vector::CUDAMutableData() method is not supported in CPU-only "
        "version."));
  }

  const T *Data(platform::Place place) const {
    PADDLE_ENFORCE_EQ(
        platform::is_cpu_place(place), true,
        platform::errors::Unavailable(
            "Vector::Data() method is not supported when not in CPUPlace."));
    return this->data();
  }

  T *MutableData(platform::Place place) {
    PADDLE_ENFORCE_EQ(
        platform::is_cpu_place(place), true,
        platform::errors::Unavailable("Vector::MutableData() method is not "
                                      "supported when not in CPUPlace."));
    return this->data();
  }

  const void *Handle() const { return static_cast<const void *>(this); }
};

template <typename T>
using Vector = CPUVector<T>;

};  // namespace framework
}  // namespace paddle
