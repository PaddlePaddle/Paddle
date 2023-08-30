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

#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard_function.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard_utils.h"

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

DistTensor::DistTensor(const phi::DenseTensor& global_value,
                       const TensorDistAttr& dist_attr)
    : dims_(global_value.dims()), dist_attr_(dist_attr), value_(global_value) {
  if (!IsDimsMappingReplicated(dist_attr_.dims_mapping())) {
    // 1. create replicated global tensor
    int64_t dims_size = global_value.dims().size();
    std::vector<int64_t> dims_mapping(dims_size, -1);
    dist_attr_.set_dims_mapping(dims_mapping);

    // 2. reshard from replicated to other state
    auto* func = ChooseProperReshardFunction(*this, dist_attr);
    auto* dev_ctx = DeviceContextPool::Instance().Get(global_value.place());
    func->Eval(dev_ctx, *this, dist_attr, this);
  }
}

DistTensor::DistTensor(const DDim& dims, const TensorDistAttr& dist_attr)
    : dims_(dims), dist_attr_(dist_attr) {}

void DistTensor::set_dims(const DDim& dims) {
  PADDLE_ENFORCE_EQ(
      this->initialized(),
      false,
      phi::errors::Unimplemented(
          "DistTensor's set_dims method can only be used when the `value` "
          "is not initialized (generally used in the InferMeta and "
          "InferSPMD stages)."));
  dims_ = dims;
}

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

bool DistTensor::defined() const { return value_.holder_ != nullptr; }

bool DistTensor::initialized() const {
  return value_.holder_ != nullptr && value_.holder_->ptr();
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
  PADDLE_THROW(phi::errors::Unavailable(
      "The DistTensor Cannot allocate memory directly and needs to perform "
      "memory operations through its DenseTensor value."));
  return nullptr;
}

}  // namespace distributed
}  // namespace phi
