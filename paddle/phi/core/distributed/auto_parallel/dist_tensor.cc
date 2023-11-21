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

#include "glog/logging.h"
#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/reshard_function.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/reshard_utils.h"
#include "paddle/phi/core/distributed/store/store_utils.h"

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

DistTensor::DistTensor() : value_(std::make_shared<DenseTensor>()) {}

DistTensor::DistTensor(const std::shared_ptr<phi::DenseTensor>& global_value,
                       const TensorDistAttr& dist_attr)
    : dims_(global_value->dims()), dist_attr_(dist_attr) {
  // If the current rank doesn't in process_mesh, we should create an
  // uninitialized tensor only with tensor_meta.
  if (IsCurRankInMesh(dist_attr.process_mesh())) {
    if (!dist_attr.is_replicated()) {
      value_ = std::make_shared<DenseTensor>();
      // 1. create replicated global tensor
      TensorDistAttr replicated_dist_attr(vectorize(global_value->dims()));
      replicated_dist_attr.set_process_mesh(dist_attr.process_mesh());
      DistTensor replicated_tensor(global_value, replicated_dist_attr);

      // 2. reshard from replicated to other state
      auto* func = ChooseProperReshardFunction(replicated_tensor, dist_attr);
      auto* dev_ctx = DeviceContextPool::Instance().Get(global_value->place());
      func->Eval(dev_ctx, replicated_tensor, dist_attr, this);
    } else {
      value_ = global_value;
    }
  } else {
    value_ = std::make_shared<DenseTensor>(
        std::make_shared<phi::Allocation>(nullptr, 0, global_value->place()),
        phi::DenseTensorMeta(global_value->meta()));
  }
}

DistTensor::DistTensor(const std::shared_ptr<phi::DenseTensor>& global_value,
                       const ProcessMesh& process_mesh,
                       const Placements& placements)
    : dims_(global_value->dims()) {
  dist_tensor_meta_ = DistTensorMeta(
      process_mesh,
      placements,
      DenseTensorMeta(global_value->dtype(), global_value->dims()));

  TensorDistAttr dist_attr(vectorize(dist_tensor_meta_.dims()));
  dist_attr.set_process_mesh(dist_tensor_meta_.process_mesh());
  dist_attr.set_dims_mapping(dist_tensor_meta_.dim_mapping());
  dist_attr.mark_annotated("process_mesh");
  dist_attr.mark_annotated("dims_mapping");
  dist_attr_ = dist_attr;

  // If the current rank doesn't in process_mesh, we should create an
  // uninitialized tensor only with dist_tensor_meta_.
  if (IsCurRankInMesh(process_mesh)) {
    if (!dist_tensor_meta_.is_replicated()) {
      value_ = std::make_shared<DenseTensor>();
      // 1. create replicated global tensor
      TensorDistAttr replicated_dist_attr(vectorize(global_value->dims()));
      replicated_dist_attr.set_process_mesh(process_mesh);
      DistTensor replicated_tensor(global_value, replicated_dist_attr);

      // 2. reshard from replicated to other state
      auto* func = ChooseProperReshardFunction(replicated_tensor, dist_attr);
      auto* dev_ctx = DeviceContextPool::Instance().Get(global_value->place());
      func->Eval(dev_ctx, replicated_tensor, dist_attr, this);
    } else {
      value_ = global_value;
    }
  } else {
    value_ = std::make_shared<DenseTensor>(
        std::make_shared<phi::Allocation>(nullptr, 0, global_value->place()),
        phi::DenseTensorMeta(global_value->meta()));
  }
}

DistTensor::DistTensor(const DDim& dims, const TensorDistAttr& dist_attr)
    : dims_(dims),
      dist_attr_(dist_attr),
      value_(std::make_shared<DenseTensor>()) {}

void DistTensor::unsafe_set_dims(const DDim& dims) {
  if (this->initialized()) {
    VLOG(3) << "You try to set an initialized DistTensor's global dims. "
               "Make sure you are aware of where you change its dims.";
  }
  dims_ = dims;
}

void DistTensor::unsafe_set_dist_attr(const TensorDistAttr& dist_attr) {
  if (this->initialized()) {
    VLOG(3) << "You try to set an initialized DistTensor's dist attr. "
               "Make sure you are aware of where you change its dist attr.";
  }
  dist_attr_ = dist_attr;
}

int64_t DistTensor::numel() const {
  // DistTensor with uninitialized local tensor can
  // also have numel.
  return product(dims_);
}

const DDim& DistTensor::local_dims() const {
  check_defined(*this, "local_dims");
  return value_->dims();
}

bool DistTensor::valid() const {
  check_defined(*this, "valid");
  return value_->valid();
}

bool DistTensor::defined() const { return value_->holder_ != nullptr; }

bool DistTensor::initialized() const {
  return value_->holder_ != nullptr && value_->holder_->ptr();
}

DataType DistTensor::dtype() const {
  // DistTensor with uninitialized local tensor can
  // also have dtype.
  return value_->dtype();
}

DataLayout DistTensor::layout() const {
  // DistTensor with uninitialized local tensor can
  // also have layout.
  return value_->layout();
}

const Place& DistTensor::place() const {
  check_defined(*this, "place");
  return value_->holder_->place();
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
