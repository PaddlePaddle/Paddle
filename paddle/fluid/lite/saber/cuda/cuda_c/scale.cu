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

#include "paddle/fluid/lite/saber/cuda/scale.h"

namespace paddle {
namespace lite {
namespace saber {
namespace cuda {

template <typename Dtype>
__global__ void ker_scale_fwd(Dtype* out_data, const int count,
                              const float scale, const float shift,
                              const Dtype* in_data) {
  CUDA_KERNEL_LOOP(tid, count) { out_data[tid] = in_data[tid] * scale + shift; }
}

class Scale : public KernelLite<TARGET(kHost), PRECISION(kFloat)> {
 public:
  using param_t = operators::MulParam;

  void Run() override {
    auto& param = Param<operators::ScaleParam>();

    auto& context = context_->As<CUDAContext>();
    int count = param.x->dims().production();
    ker_scale_fwd<float><<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0,
                           context.exec_stream>>>(
        param.output->mutable_data<float>(), count, param.scale, param.bias,
        param.x->data<float>());
  }

  virtual ~Scale() = default;
};

}  // namespace cuda
}  // namespace saber
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(mul, kCUDA, kFloat, kNCHW,
                     paddle::lite::saber::cuda::Scale, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .Finalize();
