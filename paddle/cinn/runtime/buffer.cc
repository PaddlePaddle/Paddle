// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "cinn/runtime/buffer.h"

namespace cinn {
namespace runtime {

Shape::Shape(const Shape &other)
    : data_(new value_type[other.ndims()]), ndims_(other.ndims()) {
  if (ndims() > 0) {
    memcpy(data_, other.data(), ndims_ * sizeof(value_type));
  }
}

void Shape::Resize(int ndim) {
  CHECK_GT(ndim, 0);
  ndims_ = ndim;
  if (data_) delete data_;
  data_ = new value_type[ndim];
}

Shape::value_type &Shape::operator[](int i) {
  CHECK_GT(ndims_, 0) << "shape is empty";
  CHECK_LT(i, ndims_) << "index " << i << "out of range " << ndims_;
  return data_[i];
}

Shape::value_type Shape::operator[](int i) const {
  CHECK_GT(ndims_, 0) << "shape is empty";
  CHECK_LT(i, ndims_) << "index " << i << "out of range " << ndims_;
  return data_[i];
}

uint32_t Shape::num_elements() const {
  uint32_t res = ndims_ > 0 ? 1 : 0;
  for (int i = 0; i < ndims(); i++) res *= (*this)[i];
  return res;
}

}  // namespace runtime
}  // namespace cinn
