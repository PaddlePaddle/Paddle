/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/common/int_array.h"

#include "paddle/common/ddim.h"
#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/tensor_utils.h"

namespace paddle::experimental {

template <typename T>
IntArrayBase<T>::IntArrayBase(const phi::DDim& dims) {
  AssignData(dims.Get(), dims.size());
}

template <>
IntArrayBase<phi::DenseTensor>::IntArrayBase(
    const phi::DenseTensor& tensor) {  // NOLINT
  is_from_tensor_ = true;
  if (tensor.place().GetType() == AllocationType::CPU) {
    AssignDataFromTensor(tensor);
  } else {
    phi::DenseTensor tensor_tmp;
    phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
    auto dev_ctx = pool.Get(tensor.place());
    phi::Copy(*dev_ctx, tensor, CPUPlace(), true, &tensor_tmp);
    AssignDataFromTensor(tensor_tmp);
  }
}

template <>
IntArrayBase<phi::DenseTensor>::IntArrayBase(
    const std::vector<phi::TensorRef>& tensor_ref_list) {
  is_from_tensor_ = true;
  for (auto tensor_ref : tensor_ref_list) {
    DataType data_type = tensor_ref.Get()->dtype();
    switch (data_type) {
      case DataType::INT32:
        if (tensor_ref.Get()->place().GetType() == AllocationType::CPU) {
          array_.push_back(*tensor_ref.Get()->template data<int32_t>());
        } else {
          phi::DenseTensor tensor_tmp;
          phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
          auto dev_ctx = pool.Get(tensor_ref.Get()->place());
          phi::Copy(
              *dev_ctx, *(tensor_ref.Get()), CPUPlace(), true, &tensor_tmp);
          array_.push_back(*tensor_tmp.template data<int32_t>());
        }
        break;
      case DataType::INT64:
        if (tensor_ref.Get()->place().GetType() == AllocationType::CPU) {
          array_.push_back(*tensor_ref.Get()->template data<int64_t>());
        } else {
          phi::DenseTensor tensor_tmp;
          phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
          auto dev_ctx = pool.Get(tensor_ref.Get()->place());
          phi::Copy(
              *dev_ctx, *(tensor_ref.Get()), CPUPlace(), true, &tensor_tmp);
          array_.push_back(*tensor_tmp.template data<int64_t>());
        }
        break;
      default:
        PD_THROW(
            "Data type error. Currently, The data type of IntArrayBase "
            "only supports Tensor with int32 and int64, "
            "but now received `",
            data_type,
            "`.");
    }
  }
}

template <>
IntArrayBase<phi::DenseTensor>::IntArrayBase(
    const std::vector<phi::DenseTensor>& tensor_list) {
  is_from_tensor_ = true;
  for (const auto& tensor_item : tensor_list) {
    DataType data_type = tensor_item.dtype();
    switch (data_type) {
      case DataType::INT32:
        if (tensor_item.place().GetType() == AllocationType::CPU) {
          array_.push_back(*tensor_item.template data<int32_t>());
        } else {
          phi::DenseTensor tensor_tmp;
          phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
          auto dev_ctx = pool.Get(tensor_item.place());
          phi::Copy(*dev_ctx, tensor_item, CPUPlace(), true, &tensor_tmp);
          array_.push_back(*tensor_tmp.template data<int32_t>());
        }
        break;
      case DataType::INT64:
        if (tensor_item.place().GetType() == AllocationType::CPU) {
          array_.push_back(*tensor_item.template data<int64_t>());
        } else {
          phi::DenseTensor tensor_tmp;
          phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
          auto dev_ctx = pool.Get(tensor_item.place());
          phi::Copy(*dev_ctx, tensor_item, CPUPlace(), true, &tensor_tmp);
          array_.push_back(*tensor_tmp.template data<int64_t>());
        }
        break;
      default:
        PD_THROW(
            "Data type error. Currently, The data type of IntArrayBase "
            "only supports Tensor with int32 and int64, "
            "but now received `",
            data_type,
            "`.");
    }
  }
}

}  // namespace paddle::experimental
