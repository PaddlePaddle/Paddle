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

#include <Eigen/Core>
#include "paddle/fluid/lite/core/kernel.h"
#include "paddle/fluid/lite/core/op_registry.h"
#include "paddle/fluid/lite/core/types.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

template <typename T>
void scale_compute(const T* x, T* out, int size, float scale, float bias,
                   bool bias_before) {
  if (bias_before) bias *= scale;
  for (int i = 0; i < size; i++) {
    out[i] = x[i] * scale + bias;
  }
}

class ScaleCompute : public KernelLite<TARGET(kHost), PRECISION(kFloat)> {
 public:
  using param_t = operators::MulParam;

  void Run() override {
    auto& param = Param<operators::ScaleParam>();
    scale_compute(param.x->data<float>(), param.output->mutable_data<float>(),
                  param.x->dims().production(), param.scale, param.bias,
                  param.bias_after_scale);
  }

  virtual ~ScaleCompute() = default;
};

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(scale, kHost, kFloat, kNCHW,
                     paddle::lite::kernels::host::ScaleCompute, def)
    .BindInput("X", {paddle::lite::Type::Get<paddle::lite::TensorFp32NCHWTy>(
                        TARGET(kHost))})
    .BindOutput("Out", {paddle::lite::Type::Get<paddle::lite::TensorFp32NCHWTy>(
                           TARGET(kHost))})
    .Finalize();
