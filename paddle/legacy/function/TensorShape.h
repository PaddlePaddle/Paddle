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

#include <glog/logging.h>

namespace paddle {

/**
 * TensorShape used to represent shape of normal tensor.
 */
class TensorShape {
 public:
  TensorShape() : ndims_(0), nelements_(0) { initDims(0); }

  TensorShape(size_t ndims) : ndims_(ndims), nelements_(1) { initDims(ndims); };

  TensorShape(std::initializer_list<size_t> dims) {
    ndims_ = dims.size();
    initDims(ndims_);
    dims_.assign(dims);
    numElements();
  };

  TensorShape(const TensorShape& t)
      : ndims_(t.ndims_), nelements_(t.nelements_) {
    initDims(ndims_);
    dims_.assign(t.dims_.begin(), t.dims_.end());
  };

  // get the size of specified dimension
  size_t operator[](size_t dim) const {
    CHECK_GE(dim, (size_t)0);
    CHECK_LT(dim, ndims_);
    return dims_[dim];
  }

  // set the size of specified dimension
  void setDim(size_t dim, size_t size) {
    CHECK_GE(dim, (size_t)0);
    CHECK_LT(dim, ndims_);
    dims_[dim] = size;
    numElements();
  }

  void reshape(std::initializer_list<size_t> dims) {
    ndims_ = dims.size();
    if (ndims_ > kMinDims) {
      dims_.resize(ndims_);
    }
    dims_.assign(dims);
    numElements();
  }

  // number of dimensions of the tensor
  size_t ndims() const { return ndims_; }

  size_t getElements() const { return nelements_; }

  bool operator==(const TensorShape& t) const {
    if (ndims() != t.ndims()) return false;
    for (size_t i = 0; i < ndims(); i++) {
      if (dims_[i] != t.dims_[i]) return false;
    }

    return true;
  }

  bool operator!=(const TensorShape& t) const { return !(*this == t); }

 private:
  // compute number of elements
  void numElements() {
    nelements_ = 1;
    for (size_t n = 0; n < ndims_; n++) {
      nelements_ *= dims_[n];
    }
  }

  // init dims_
  void initDims(size_t ndims) {
    size_t count = ndims < kMinDims ? kMinDims : ndims;
    dims_.assign(count, 1);
  }

  // number of dimensions
  // ndims_ may be not equeal dims_.size()
  size_t ndims_;
  // number of elements
  size_t nelements_;
  std::vector<size_t> dims_;
  static const size_t kMinDims = 4;
};

}  // namespace paddle
