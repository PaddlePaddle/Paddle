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

#include "paddle/phi/common/tensor_ref.h"

#include <cstdint>
#include <limits>
#include <sstream>
#include <vector>

#include "paddle/phi/api/ext/exception.h"
#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/enforce.h"

namespace phi {

std::vector<int64_t> ConvertTensorRefVec2IntArray(
    const std::vector<TensorRef>& tensor_ref_list) {
  std::vector<int64_t> vec_res;
  vec_res.reserve(tensor_ref_list.size());

  for (size_t i = 0; i < tensor_ref_list.size(); ++i) {
    DataType data_type = tensor_ref_list[i].Get()->dtype();
    switch (data_type) {
      case DataType::INT32:
        if (tensor_ref_list[i].Get()->place().GetType() ==
            AllocationType::CPU) {
          vec_res.push_back(
              *tensor_ref_list[i].Get()->template data<int32_t>());
        } else {
          phi::DenseTensor tensor_tmp;
          phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
          auto dev_ctx = pool.Get(tensor_ref_list[i].Get()->place());
          phi::Copy(*dev_ctx,
                    *(tensor_ref_list[i].Get()),
                    CPUPlace(),
                    true,
                    &tensor_tmp);
          vec_res.push_back(*tensor_tmp.template data<int32_t>());
        }
        break;
      case DataType::INT64:
        if (tensor_ref_list[i].Get()->place().GetType() ==
            AllocationType::CPU) {
          vec_res.push_back(
              *tensor_ref_list[i].Get()->template data<int64_t>());
        } else {
          phi::DenseTensor tensor_tmp;
          phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
          auto dev_ctx = pool.Get(tensor_ref_list[i].Get()->place());
          phi::Copy(*dev_ctx,
                    *(tensor_ref_list[i].Get()),
                    CPUPlace(),
                    true,
                    &tensor_tmp);
          vec_res.push_back(*tensor_tmp.template data<int64_t>());
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

  return std::move(vec_res);
}

}  // namespace phi
