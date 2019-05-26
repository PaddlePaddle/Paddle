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
  CHECK_EQ(param.bias->numel(), w_dims[1]);
  CHECK_EQ(param.output->dims().size(), 2UL);
#ifdef __aarch64__
  LOG(INFO) << "--------64-";
#else
  LOG(INFO) << "--------32-";
#endif
  // ARMContext ctx;

  int x_h = x_dims.Slice(0, param.in_num_col_dims).production();
  if (x_h > 1) {
    // float *pre_din = static_cast<float *>(this->_ctx->get_work_space()) +
    // this->_ctx->l2_cache_size() / sizeof(float);
    // prepackA(pre_din, (const float*)din, _k, 0, _m, 0, _k, false,
    // this->_ctx);
    // sgemm_prepack(pre_din, (const float*)weights, (const float*)bias,
    // (float*)dout, _m, _n, _k, false, false,
    //               !_param->_flag_trans, this->_ctx);
    // if (_param->_flag_bias) {
    //     fill_bias_fc((float*)dout, (const float*)bias, _m, _n);
    // }
  } else {
    // use sgemmv
    //            sgemv((const float*)weights, (const float*)din, (float*)dout,
    //            false, _n, _k, _param->_flag_bias, (float*)bias, false);
  }

  fc_compute_eigen(param.input->data<float>(),  // x
                   x_h,
                   param.input->dims()
                       .Slice(param.in_num_col_dims, param.input->dims().size())
                       .production(),
                   param.w->data<float>(),     // w
                   w_dims[0],                  // w_h
                   w_dims[1],                  // w_w
                   param.bias->data<float>(),  // b
                   param.output->mutable_data<float>());
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
