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

#include "paddle/fluid/lite/kernels/host/fc_compute.h"
#include <Eigen/Core>
#include "paddle/fluid/lite/core/op_registry.h"
#include "paddle/fluid/lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

// NOTE should use pure std C++ implementation.
void FcCompute::Run() {
  auto& param = this->Param<operators::FcParam>();

  CHECK_GE(param.input->dims().size(), 2UL);
  CHECK_EQ(param.output->dims().size(), 2UL);

  fc_compute_eigen(
      param.input->data<float>(),  // x
      param.input->dims().Slice(0, param.in_num_col_dims).production(),
      param.input->dims()
          .Slice(param.in_num_col_dims, param.input->dims().size())
          .production(),
      param.w->data<float>(),     // w
      param.w->dims()[1],         // w_w
      param.w->dims()[0],         // w_h
      param.bias->data<float>(),  // b
      param.output->mutable_data<float>());
}

// TargetType FcCompute::target() const { return TARGET(kHost); }

// PrecisionType FcCompute::precision() const { return PRECISION(kFloat); }

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(fc, kHost, kFloat, kNCHW,
                     paddle::lite::kernels::host::FcCompute, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("W", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();
