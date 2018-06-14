/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#define EIGEN_USE_GPU

#include "paddle/fluid/operators/triplet_loss_op.h"

namespace paddle {
namespace operators {

std::vector<int> GetOffsets<platform::CUDADeviceContext>(const Tensor* t) {
  TensorCopy t_cpu;
  framework::TensorCopySync(*t, platform::CPUPlace(), t_cpu);
  std::vector<int> offsets;
  int64_t* data = t_cpu->data<int64_t>();
  int offset = 0;
  int64_t currrent_value = data[0];
  for (int i = 1; i < t->numel(); ++i) {
    if (data[i] != currrent_value) {
      offsets.push(i);
    }
    currrent_value = data[i];
  }
  offsets.push(t->numel());
  return offsets;
}

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    triplet_loss, ops::TripletLossKernel<platform::CUDADeviceContext, float>,
    ops::TripletLossKernel<platform::CUDADeviceContext, double>);
REGISTER_OP_CUDA_KERNEL(
    triplet_loss_grad,
    ops::TripletLossGradKernel<platform::CUDADeviceContext, float>,
    ops::TripletLossGradKernel<platform::CUDADeviceContext, double>);
