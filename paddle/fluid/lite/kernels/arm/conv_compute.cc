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

#include "paddle/fluid/lite/kernels/arm/conv_compute.h"
#include "paddle/fluid/lite/arm/math/conv_direct.h"
#include "paddle/fluid/lite/arm/math/conv_depthwise.h"
#include "paddle/fluid/lite/arm/math/conv_gemmlike.h"
#include "paddle/fluid/lite/arm/math/funcs.h"
#include "paddle/fluid/lite/core/op_registry.h"
#include "paddle/fluid/lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

void ConvCompute::Run() {
  auto& param = this->Param<param_t>();
  auto x_dims = param.x->dims();
  auto w_dims = param.filter->dims();
  auto o_dims = param.output->dims();

  auto& ctx = this->ctx_->template As<ARMContext>();

  int win = x_dims[3];  // nchw
  int hin = x_dims[2];
  int ic = x_dims[1];
  int bs = x_dims[0];
  int ow = o_dims[3];
  int oh = o_dims[2];
  int oc = o_dims[1];
  int kh = w_dims[2];  // oihw
  int kw = w_dims[3];
  int pad = param.paddings[0];
  int stride = param.strides[0];

  const auto* i_data = param.x->data<float>();
  const auto* w_data = param.filter->data<float>();
  const auto* b_data = param.bias ? param.bias->data<float>() : nullptr;
  auto* o_data = param.output->mutable_data<float>();

  // TODO(xxx): create should be somewhere better!
  bool kps_equal = (param.paddings[0] == param.paddings[1]) &&
                   (param.strides[0] == param.strides[1]) && (kw == kh);
  bool no_dilation = (param.dilations[0] == 1) && (param.dilations[1] == 1);
  bool flag_dw_3x3 =
      (kw == 3 && (pad == 0 || pad == 1) && (stride == 1 || stride == 2));
  bool flag_dw_5x5 =
      (kw == 5 && stride == 1) || (kw == 5 && stride == 2 && pad == 2);
  bool flag_dw = flag_dw_3x3 || flag_dw_5x5;

  // select conv impl
  // TODO(xxx): enable more
  if (param.groups == ic && ic == oc && kps_equal && no_dilation && flag_dw) {
    // dw conv impl
    impl_ = new lite::arm::math::DepthwiseConv<PRECISION(kFloat)>;
    LOG(INFO) << "invoking dw conv";
  } else if (param.groups == 1 && kw == 3 && stride == 1 && kps_equal &&
             no_dilation) {
    if (ic >= 32 && oc >= 32 && oh > 16 && ow > 16) {
      // winograd conv impl
      // impl_ = new lite::arm::math::WinogradConv<PRECISION(kFloat)>;
      LOG(FATAL) << "TODO!!! winograd conv";
    } else {
      // direct conv impl
      impl_ = new lite::arm::math::DirectConv<PRECISION(kFloat)>;
      LOG(INFO) << "invoking direct conv";
    }
  } else if (param.groups == 1 && kw == 3 && stride == 2 && kps_equal &&
             no_dilation) {
    // direct conv impl
    impl_ = new lite::arm::math::DirectConv<PRECISION(kFloat)>;
  } else {
    impl_ = new lite::arm::math::GemmLikeConv<PRECISION(kFloat)>;
    LOG(INFO) << "invoking gemm like conv";
  }
  this->impl_->create(param, &ctx);

  CHECK(impl_);
  impl_->run(param);

  // if (this->act_ != nullptr) {
  //   this->act_->run(outputs, outputs, param.activation_param);
  // }
}

TargetType ConvCompute::target() const { return TARGET(kARM); }

PrecisionType ConvCompute::precision() const { return PRECISION(kFloat); }

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(conv2d, kARM, kFloat, kNCHW,
                     paddle::lite::kernels::arm::ConvCompute, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();

REGISTER_LITE_KERNEL(depthwise_conv2d, kARM, kFloat, kNCHW,
                     paddle::lite::kernels::arm::ConvCompute, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
