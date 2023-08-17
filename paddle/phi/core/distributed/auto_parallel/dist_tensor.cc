// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/distributed/auto_parallel/dist_tensor.h"

namespace phi {
namespace distributed {

inline void check_defined(const DistTensor& dist_tensor,
                          std::string method_hint) {
  PADDLE_ENFORCE_EQ(
      dist_tensor.defined(),
      true,
      phi::errors::Unimplemented(
          "DistTensor is not defined yet when `%s` method is called.",
          method_hint));
}

// TODO(chenweihang): Reshard the input global value into local value
DistTensor::DistTensor(const phi::DenseTensor& global_value,
                       const TensorDistAttr& dist_attr)
    : dims_(global_value.dims()), dist_attr_(dist_attr), value_(global_value) {}

DistTensor::DistTensor(const phi::DenseTensor& value,
                       const DDim& dims,
                       const TensorDistAttr& dist_attr)
    : dims_(dims), dist_attr_(dist_attr), value_(value) {}

DistTensor::DistTensor(const DDim& dims, const TensorDistAttr& dist_attr)
    : dims_(dims), dist_attr_(dist_attr) {}

void DistTensor::set_dims(const DDim& dims) { dims_ = dims; }

int64_t DistTensor::numel() const {
  check_defined(*this, "numel");
  return value_.numel();
}

const DDim& DistTensor::local_dims() const {
  check_defined(*this, "local_dims");
  return value_.dims();
}

bool DistTensor::valid() const {
  check_defined(*this, "valid");
  return value_.valid();
}

DataType DistTensor::dtype() const {
  check_defined(*this, "dtype");
  return value_.dtype();
}

DataLayout DistTensor::layout() const {
  check_defined(*this, "layout");
  return value_.layout();
}

const Place& DistTensor::place() const {
  check_defined(*this, "place");
  return value_.holder_->place();
}

void* DistTensor::AllocateFrom(Allocator* allocator,
                               DataType dtype,
                               size_t requested_size,
                               bool fake_alloc) {
  return value_.AllocateFrom(allocator, dtype, requested_size, fake_alloc);
}

}  // namespace distributed
}  // namespace phi
