// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/cudnn_helper.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

struct CudnnDropoutCache {
  CudnnDropoutCache() {
    PADDLE_ENFORCE(platform::dynload::cudnnCreateTensorDescriptor(&data_desc_));
    PADDLE_ENFORCE(
        platform::dynload::cudnnCreateDropoutDescriptor(&dropout_desc_));
  }
  ~CudnnDropoutCache() {
    PADDLE_ENFORCE(platform::dynload::cudnnDestroyTensorDescriptor(data_desc_));
    PADDLE_ENFORCE(
        platform::dynload::cudnnDestroyDropoutDescriptor(dropout_desc_));
  }

  cudnnTensorDescriptor_t data_desc_;
  cudnnDropoutDescriptor_t dropout_desc_;
  size_t states_size_in_bytes_, reserve_space_size_in_bytes_;
  int64_t input_size_ = -1;
  Tensor states_;  // dtype = uint8
};
}  // namespace operators
}  // namespace paddle
