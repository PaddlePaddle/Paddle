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
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/temporay_allocator.h"
namespace paddle {
namespace platform {

template <typename T>
paddle::framework::Tensor GetTensor(
    memory::allocation::AllocationPtr temp_allocation_ptr,
    const framework::DDim &dim) {
  auto &deleter = temp_allocation_ptr.get_deleter();
  auto *allocation_ptr = temp_allocation_ptr.release();
  auto shared_allocation =
      std::shared_ptr<memory::allocation::Allocation>(allocation_ptr, deleter);

  PADDLE_ENFORCE(dynamic_cast<TemporayAllocation *>(allocation_ptr) != nullptr,
                 "The AllocationPtr must be TemporayAllocation.");
  PADDLE_ENFORCE_EQ(allocation_ptr->size(),
                    framework::product(dim) * sizeof(T));

  paddle::framework::Tensor temp_tensor(std::type_index(typeid(T)));
  temp_tensor.Resize(dim);
  temp_tensor.ReSetHolder(std::move(shared_allocation));
  return temp_tensor;
}

}  // namespace platform
}  // namespace paddle
