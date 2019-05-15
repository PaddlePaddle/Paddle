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

#pragma once
#include "paddle/fluid/lite/core/kernel.h"
#include "paddle/fluid/lite/core/op_registry.h"
#include "paddle/fluid/lite/cuda/cuda_utils.h"

namespace paddle {
namespace lite {
namespace saber {
namespace cuda {

class ScaleCompute : public KernelLite<TARGET(kHost), PRECISION(kFloat)> {
 public:
  using param_t = operators::MulParam;

  void Run() override {
    auto& param = Param<operators::ScaleParam>();

    // TODO(Shixiaowei02): Assign value to cudastream.
    cudaStream_t cuda_stream;
    int count = param.x->dims().production();
    ker_power_fwd<
        float><<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(
        param.output->mutable_data<float>(), count, param.scale, param.shift,
        param.x->data<float>());
  }

  virtual ~ScaleCompute() = default;
};

}  // namespace cuda
}  // namespace saber
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(scale, kHost, kFloat, kNCHW,
                     paddle::lite::saber::cuda::ScaleCompute, def)
    .BindInput("X", {paddle::lite::Type::Get<paddle::lite::TensorFp32NCHWTy>(
                        TARGET(kHost))})
    .BindOutput("Out", {paddle::lite::Type::Get<paddle::lite::TensorFp32NCHWTy>(
                           TARGET(kHost))})
    .Finalize();
