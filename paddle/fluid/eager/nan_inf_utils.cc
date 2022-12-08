// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/eager/nan_inf_utils.h"

#include "paddle/fluid/framework/details/nan_inf_utils_detail.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/selected_rows.h"

namespace egr {

void CheckTensorHasNanOrInf(const std::string& api_name, const Tensor& tensor) {
  if (tensor.initialized()) {
    auto& tensor_name = tensor.name();
    const phi::DenseTensor* dense_tensor{nullptr};
    if (tensor.is_dense_tensor()) {
      dense_tensor = static_cast<const phi::DenseTensor*>(tensor.impl().get());
    } else if (tensor.is_selected_rows()) {
      dense_tensor = &(
          static_cast<const phi::SelectedRows*>(tensor.impl().get())->value());
    } else {
      VLOG(10) << "Only DenseTensor or SelectedRows need to check, "
               << tensor_name << " is no need.";
      return;
    }

    auto& place = dense_tensor->place();
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      paddle::framework::details::tensor_check<phi::GPUContext>(
          api_name, tensor_name, *dense_tensor, place);
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "Tensor[%s] use gpu place. PaddlePaddle must compile with GPU.",
          tensor_name));
#endif
      return;
    }
    paddle::framework::details::tensor_check<phi::CPUContext>(
        api_name, tensor_name, *dense_tensor, place);
  }
}

void CheckTensorHasNanOrInf(const std::string& api_name,
                            const TupleOfTwoTensors& tensors) {
  CheckTensorHasNanOrInf(api_name, std::get<0>(tensors));
  CheckTensorHasNanOrInf(api_name, std::get<1>(tensors));
}

void CheckTensorHasNanOrInf(const std::string& api_name,
                            const TupleOfThreeTensors& tensors) {
  CheckTensorHasNanOrInf(api_name, std::get<0>(tensors));
  CheckTensorHasNanOrInf(api_name, std::get<1>(tensors));
  CheckTensorHasNanOrInf(api_name, std::get<2>(tensors));
}

void CheckTensorHasNanOrInf(const std::string& api_name,
                            const TupleOfFourTensors& tensors) {
  CheckTensorHasNanOrInf(api_name, std::get<0>(tensors));
  CheckTensorHasNanOrInf(api_name, std::get<1>(tensors));
  CheckTensorHasNanOrInf(api_name, std::get<2>(tensors));
  CheckTensorHasNanOrInf(api_name, std::get<3>(tensors));
}

void CheckTensorHasNanOrInf(const std::string& api_name,
                            const TupleOfFiveTensors& tensors) {
  CheckTensorHasNanOrInf(api_name, std::get<0>(tensors));
  CheckTensorHasNanOrInf(api_name, std::get<1>(tensors));
  CheckTensorHasNanOrInf(api_name, std::get<2>(tensors));
  CheckTensorHasNanOrInf(api_name, std::get<3>(tensors));
  CheckTensorHasNanOrInf(api_name, std::get<4>(tensors));
}

void CheckTensorHasNanOrInf(const std::string& api_name,
                            const TupleOfSixTensors& tensors) {
  CheckTensorHasNanOrInf(api_name, std::get<0>(tensors));
  CheckTensorHasNanOrInf(api_name, std::get<1>(tensors));
  CheckTensorHasNanOrInf(api_name, std::get<2>(tensors));
  CheckTensorHasNanOrInf(api_name, std::get<3>(tensors));
  CheckTensorHasNanOrInf(api_name, std::get<4>(tensors));
  CheckTensorHasNanOrInf(api_name, std::get<5>(tensors));
}

void CheckTensorHasNanOrInf(const std::string& api_name,
                            const std::vector<Tensor>& tensors) {
  for (auto& tensor : tensors) {
    CheckTensorHasNanOrInf(api_name, tensor);
  }
}

void CheckTensorHasNanOrInf(
    const std::string& api_name,
    const paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>& tensors) {
  for (auto& tensor_vector : tensors) {
    CheckTensorHasNanOrInf(api_name, tensor_vector);
  }
}

void CheckTensorHasNanOrInf(const std::string& api_name,
                            const TupleOfTensorAndVector& tensors) {
  CheckTensorHasNanOrInf(api_name, std::get<0>(tensors));
  CheckTensorHasNanOrInf(api_name, std::get<1>(tensors));
  CheckTensorHasNanOrInf(api_name, std::get<2>(tensors));
}

}  // namespace egr
