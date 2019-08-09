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

/*
 * This file defines the general interface for DDim and Tensor, which is used in
 * server and mobile framework, to make the framework on the two devices share
 * the same code, we clear up the methods and make the different implementations
 * looks the same.
 */

#include <string>
#include <vector>
#include "paddle/fluid/lite/core/target_wrapper.h"

namespace paddle {
namespace lite {

/*
 * This class defines the basic interfaces of the DDims for server and mobile.
 * For the DDims's implementation is too tedious, we add a simple implementation
 * for mobile, and use this interface to share the framework both for mobile and
 * server.
 *
 * The derived should implement following interfaces:
 * ConstructFrom
 * operator[]
 * Vectorize
 * size
 */
template <typename DDimT>
class DDimBase {
 public:
  using value_type = int64_t;

  DDimBase() = default;

  explicit DDimBase(const std::vector<int64_t> &x) { self()->ConstructFrom(x); }
  value_type operator[](int offset) const { return (*const_self())[offset]; }
  value_type &operator[](int offset) { return (*self())[offset]; }
  std::vector<int64_t> Vectorize() const { return self()->Vectorize(); }
  size_t size() const { return const_self()->size(); }
  bool empty() const { return const_self()->empty(); }

  value_type production() const {
    value_type res = 1;
    for (size_t i = 0; i < const_self()->size(); i++) {
      res *= (*const_self())[i];
    }
    return res;
  }

  DDimT Slice(int start, int end) const {
    std::vector<value_type> vec;
    for (int i = start; i < end; i++) {
      vec.push_back((*const_self())[i]);
    }
    return DDimT(vec);
  }

  DDimT Flattern2D(int col) const {
    return DDimT(std::vector<value_type>(
        {Slice(0, col).production(), Slice(col, size()).production()}));
  }

  std::string repr() const {
    std::stringstream ss;
    ss << "{";
    for (size_t i = 0; i < this->size() - 1; i++) {
      ss << (*this)[i] << ",";
    }
    if (!this->empty()) ss << (*this)[size() - 1];
    ss << "}";
    return ss.str();
  }

  friend std::ostream &operator<<(std::ostream &os, const DDimT &dims) {
    os << dims.repr();
    return os;
  }

  friend bool operator==(const DDimBase &a, const DDimBase &b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++) {
      if (a[i] != b[i]) return false;
    }
    return true;
  }

  friend bool operator!=(const DDimBase &a, const DDimBase &b) {
    return !(a == b);
  }

 private:
  DDimT *self() { return static_cast<DDimT *>(this); }
  const DDimT *const_self() const { return static_cast<const DDimT *>(this); }
};

/*
 * This class defines the basic interfaces of the tensors implemented for
 * server and mobile. It use the CRTR technology to accelerate the runtime
 * performance.
 */
template <typename TensorT>
class TensorBase {
 public:
  TensorBase() = default;

  template <typename T, typename DimT>
  void Assign(T *data, const DimT &dim) {
    self()->Assign(data, dim);
  }

  TargetType target() const { return self()->target(); }

  template <typename T>
  T *mutable_data() {
    return self()->template mutable_data<T>();
  }

  template <typename T>
  T *mutable_data(TargetType target) {
    return self()->template mutable_data<T>(target);
  }

  template <typename T>
  const T *data() {
    return self()->template data<T>();
  }

  template <typename DimT>
  void Resize(const DimT &dims) {
    self()->Resize(dims);
  }

  template <typename DDimT>
  DDimT dims() {
    return self()->dims();
  }

  template <typename LoDT>
  const LoDT &lod() const {
    return const_self()->lod();
  }
  template <typename LoDT>
  LoDT *mutable_lod() {
    return self()->mutable_lod();
  }
  template <typename T>
  const T &data() const {
    return const_self()->data();
  }

  const void *raw_data() const { return const_self()->data(); }

  size_t data_size() const { return const_self()->dims().production(); }
  size_t memory_size() const { return const_self()->memory_size(); }

  void ShareDataWith(const TensorBase &other) { self()->ShareDataWith(other); }
  void CopyDataFrom(const TensorBase &other) { self()->CopyDataFrom(other); }

  friend std::ostream &operator<<(std::ostream &os, const TensorT &tensor) {
    os << "Tensor:" << '\n';
    os << "dim: " << tensor.dims() << '\n';
    for (int i = 0; i < tensor.dims().production(); i++) {
      os << tensor.template data<float>()[i] << " ";
    }
    os << "\n";
    return os;
  }

 private:
  TensorT *self() { return static_cast<TensorT *>(this); }
  const TensorT *const_self() const {
    return static_cast<const TensorT *>(this);
  }
};

template <typename TensorT>
bool TensorCompareWith(const TensorT &a, const TensorT &b) {
  if (a.dims() != b.dims()) return false;
  if (memcmp(a.raw_data(), b.raw_data(), a.data_size()) != 0) return false;
  return true;
}

}  // namespace lite
}  // namespace paddle
