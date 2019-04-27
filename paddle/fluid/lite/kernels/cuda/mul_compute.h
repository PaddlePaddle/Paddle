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
#include "paddle/fluid/lite/core/types.h"
#include "paddle/fluid/lite/cuda/blas.h"
#include "paddle/fluid/lite/operators/op_params.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

template <typename T>
void mul_compute(const lite::cuda::Blas<float>& blas, const T* x, int x_h,
                 int x_w, const T* y, int y_h, int y_w, T* out) {
  blas.sgemm(CUBLAS_OP_N, CUBLAS_OP_N, x_h, y_w, x_w, nullptr, x, x_w, y, y_w,
             nullptr, out, x_h);
}

class MulCompute : public OpKernel<TARGET(kCUDA), PRECISION(kFloat)> {
 public:
  using param_t = operators::MulParam;

  void Run() override {
    CHECK(context_) << "running context should be set first";
    auto& context = context_->AsCudaContext();
    CHECK(context.blas_fp32) << "blas should init first";
    /*
    auto& blas = *context.blas_fp32;
    CHECK(param.x->target() == TARGET(kCUDA));
    auto* x = param.x->data<float>();
    int x_h = param.x->dims()[0];
    int x_w = param.x->dims()[1];

    auto* y = param.y->data<float>();
    int y_h = param.y->dims()[0];
    int y_w = param.y->dims()[1];
     */

    const auto& param = Param<operators::MulParam>();
    param.output->mutable_data<float>(TARGET(kCUDA));
    LOG(INFO) << "mul output memory size " << param.output->memory_size();

    // mul_compute<float>(blas, x, x_h, x_w, y, y_h, y_w, out);
  }

  virtual ~MulCompute() = default;
};

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
