//   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/pull_gpups_sparse_op.h"
#include "paddle/fluid/platform/device/gpu/gpu_info.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"

namespace paddle {
namespace operators {
using platform::PADDLE_CUDA_NUM_THREADS;
using LoDTensor = framework::LoDTensor;

template <typename T>
class PullGpuPSSparseCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PullGpuPSSparseFunctor<T>(ctx);
  }
};

template <typename T>
class PushGpuPSSparseCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PushGpuPSSparseFunctor<T>(ctx);
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(pull_gpups_sparse,
                        ops::PullGpuPSSparseCUDAKernel<float>,
                        ops::PullGpuPSSparseCUDAKernel<double>)
REGISTER_OP_CUDA_KERNEL(push_gpups_sparse,
                        ops::PushGpuPSSparseCUDAKernel<float>,
                        ops::PushGpuPSSparseCUDAKernel<double>)