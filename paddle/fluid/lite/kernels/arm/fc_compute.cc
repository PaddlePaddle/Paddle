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

#include "paddle/fluid/lite/kernels/arm/fc_compute.h"
#include "paddle/fluid/lite/core/op_registry.h"
#include "paddle/fluid/lite/core/type_system.h"
#include "paddle/fluid/lite/kernels/arm/packed_sgemm.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

// NOTE should use pure std C++ implementation.
void FcCompute::Run() {
  auto& param = this->Param<operators::FcParam>();
  auto x_dims = param.input->dims();
  auto w_dims = param.w->dims();

  CHECK_GE(x_dims.size(), 2UL);
  CHECK_EQ(w_dims.size(), 2UL);

  CHECK_EQ(param.output->dims().size(), 2UL);
  const auto* i_data = param.input->data<float>();
  const auto* w_data = param.w->data<float>();
  const auto* b_data = param.bias ? param.bias->data<float>() : nullptr;
  auto* o_data = param.output->mutable_data<float>();

  int x_h = x_dims.Slice(0, param.in_num_col_dims).production();
  int x_w = x_dims.Slice(param.in_num_col_dims, x_dims.size()).production();
  int n = w_dims[1];
  CHECK_EQ(x_w, static_cast<int>(w_dims[0]));
  auto& ctx = this->ctx_->template As<ARMContext>();
  if (x_h > 1) {
    float* packed_in = static_cast<float*>(ctx.get_workspace_data<float>()) +
                       ctx.l2_cache_size() / sizeof(float);
    prepackA(packed_in, i_data, x_w, 0, x_h, 0, x_w, false, &ctx);
    sgemm_prepack(packed_in, w_data, b_data, o_data, x_h, n, x_w, false, false,
                  false, &ctx);

    if (param.bias) {
      CHECK_EQ(param.bias->numel(), n);
      fill_bias_fc(o_data, b_data, x_h, n);
    }
  } else {
    // use sgemmv
    // sgemv((const float*)weights, (const float*)din, (float*)dout,
    //       false, n, x_w, _param->_flag_bias, (float*)bias, false);
  }
}

TargetType FcCompute::target() const { return TARGET(kARM); }

PrecisionType FcCompute::precision() const { return PRECISION(kFloat); }

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(fc, kARM, kFloat, kNCHW,
                     paddle::lite::kernels::arm::FcCompute, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("W", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
