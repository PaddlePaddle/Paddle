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

#include "paddle/fluid/lite/kernels/arm/elementwise_add_compute.h"
#include "paddle/fluid/lite/arm/math/funcs.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

void ElementwiseAddCompute::Run() {
  auto& param = Param<operators::ElementwiseParam>();
  const float* x_data = param.X->data<float>();
  const float* y_data = param.Y->data<float>();
  float* out_data = param.Out->mutable_data<float>();
  int axis = param.axis;
  auto x_dims = param.X->dims();
  auto y_dims = param.Y->dims();
  if (axis < 0) {
    axis = x_dims.size() - y_dims.size();
  }
  if (x_dims.size() == y_dims.size()) {
    lite::arm::math::elementwise_add(x_data, y_data, out_data,
                                     x_dims.production());
  } else {
    int batch = 1;
    int channels = 1;
    int num = 1;
    for (int i = 0; i < axis; ++i) {
      batch *= x_dims[i];
    }
    for (int i = 0; i < y_dims.size(); ++i) {
      channels *= y_dims[i];
    }
    for (int i = y_dims.size() + axis; i < x_dims.size(); ++i) {
      num *= x_dims[i];
    }
    lite::arm::math::elementwise_add_axis(x_data, y_data, out_data, batch,
                                          channels, num);
  }
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(elementwise_add, kARM, kFloat, kNCHW,
                     paddle::lite::kernels::arm::ElementwiseAddCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
