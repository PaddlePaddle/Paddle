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

#pragma once

#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/tensor_util.h"
<<<<<<< HEAD
#include "paddle/phi/kernels/funcs/selected_rows_functor.h"
=======
#include "paddle/fluid/operators/math/selected_rows_functor.h"
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

namespace paddle {
namespace operators {

<<<<<<< HEAD
namespace scatter = phi::funcs::scatter;

static inline float GetAttrFromTensor(const phi::DenseTensor* tensor) {
  const float* tensor_data = tensor->data<float>();
  phi::DenseTensor cpu_tensor;
=======
namespace scatter = paddle::operators::math::scatter;

static inline float GetAttrFromTensor(const framework::Tensor* tensor) {
  const float* tensor_data = tensor->data<float>();
  framework::Tensor cpu_tensor;
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
  if (platform::is_gpu_place(tensor->place())) {
    paddle::framework::TensorCopySync(
        *tensor, platform::CPUPlace(), &cpu_tensor);
    tensor_data = cpu_tensor.data<float>();
  }
  if (platform::is_xpu_place(tensor->place())) {
    paddle::framework::TensorCopySync(
        *tensor, platform::CPUPlace(), &cpu_tensor);
    tensor_data = cpu_tensor.data<float>();
  }
  return tensor_data[0];
}

}  // namespace operators
}  // namespace paddle
