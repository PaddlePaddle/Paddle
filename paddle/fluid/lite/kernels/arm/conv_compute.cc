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
#include "paddle/fluid/lite/core/op_registry.h"
#include "paddle/fluid/lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

void ConvCompute::PrepareForRun() {
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

  bool kps_equal = (param.paddings[0] == param.paddings[1]) &&
                   (param.strides[0] == param.strides[1]) && (kw == kh);
  bool no_dilation = (param.dilations[0] == 1) && (param.dilations[1] == 1);
  bool flag_dw_3x3 =
      (kw == 3 && (pad == 0 || pad == 1) && (stride == 1 || stride == 2));
  bool flag_dw_5x5 =
      (kw == 5 && stride == 1) || (kw == 5 && stride == 2 && pad == 2);
  bool flag_dw = flag_dw_3x3 || flag_dw_5x5;

  // select conv impl
  if (param.groups == ic && ic == oc && kps_equal && no_dilation && flag_dw) {
    // dw conv impl
    impl_ = new lite::arm::math::DepthwiseConv<PRECISION(kFloat)>;
    VLOG(3) << "invoking dw conv";
  } else if (param.groups == 1 && kw == 3 && stride == 1 && kps_equal &&
             no_dilation) {
    if (ic >= 32 && oc >= 32 && oh > 16 && ow > 16) {
      // winograd conv impl
      impl_ = new lite::arm::math::WinogradConv<PRECISION(kFloat)>;
      VLOG(3) << "invoking winograd conv";
    } else {
      // direct conv impl
      impl_ = new lite::arm::math::DirectConv<PRECISION(kFloat)>;
      VLOG(3) << "invoking direct conv";
    }
  } else if (param.groups == 1 && kw == 3 && stride == 2 && kps_equal &&
             no_dilation) {
    // direct conv impl
    impl_ = new lite::arm::math::GemmLikeConv<PRECISION(kFloat)>;
    VLOG(3) << "invoking direct conv";
  } else {
    impl_ = new lite::arm::math::GemmLikeConv<PRECISION(kFloat)>;
    VLOG(3) << "invoking gemm like conv";
  }
  CHECK(this->impl_->create(param, &ctx));
}

void ConvCompute::Run() {
  auto& param = this->Param<param_t>();
  CHECK(impl_);
  impl_->run(param);
  // if (this->act_ != nullptr) {
  //   this->act_->run(outputs, outputs, param.activation_param);
  // }
}

template <PrecisionType Ptype_out>
void ConvComputeInt8<Ptype_out>::PrepareForRun() {
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
  int ph = param.paddings[1];
  int pw = param.paddings[0];
  int sh = param.strides[1];
  int sw = param.strides[0];

  bool kps_equal = (pw == ph) && (sh == sw) && (kw == kh);
  bool no_dilation = (param.dilations[0] == 1) && (param.dilations[1] == 1);
  bool flag_dw_3x3 = (kw == 3) && (ph == 1) && (sw == 1 || sw == 2);
  bool flag_dw_5x5 = (kw == 5 && sw == 1 && ph == 2);
  bool flag_dw = flag_dw_3x3 || flag_dw_5x5;

  // weigth is int8 and bias is int32 so do not need trans
  if (param.groups == ic && ic == oc && kps_equal && no_dilation && flag_dw) {
    impl_ = new lite::arm::math::DepthwiseConvInt8<Ptype_out>;
    VLOG(3) << "Run DepthwiseConv Int8";
  } else if (param.groups == 1 && kw == 3 && (sw == 1 || sw == 2) &&
             kps_equal && no_dilation) {
    VLOG(3) << "Run DirectConv Int8";
    impl_ = new lite::arm::math::GemmLikeConvInt8<Ptype_out>;
    // impl_ = new lite::arm::math::DirectConvInt8<Ptype_out>;
  } else {
    VLOG(3) << "Run GemmLikeConvInt8";
    impl_ = new lite::arm::math::GemmLikeConvInt8<Ptype_out>;
  }

  CHECK(this->impl_->create(param, &ctx));
}

template <PrecisionType Ptype_out>
void ConvComputeInt8<Ptype_out>::Run() {
  auto& param = this->Param<param_t>();
  CHECK(impl_);
  impl_->run(param);
}

template class ConvComputeInt8<PRECISION(kInt8)>;
template class ConvComputeInt8<PRECISION(kFloat)>;
template class ConvComputeInt8<PRECISION(kInt32)>;

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(conv2d, kARM, kFloat, kNCHW,
                     paddle::lite::kernels::arm::ConvCompute, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();

REGISTER_LITE_KERNEL(depthwise_conv2d, kARM, kFloat, kNCHW,
                     paddle::lite::kernels::arm::ConvCompute, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();

REGISTER_LITE_KERNEL(
    conv2d, kARM, kInt8, kNCHW,
    paddle::lite::kernels::arm::ConvComputeInt8<PRECISION(kInt8)>, int8_out)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt8))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("Filter",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt8))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt8))})
    .Finalize();

REGISTER_LITE_KERNEL(
    conv2d, kARM, kInt8, kNCHW,
    paddle::lite::kernels::arm::ConvComputeInt8<PRECISION(kFloat)>, fp32_out)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt8))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("Filter",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt8))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .Finalize();

REGISTER_LITE_KERNEL(
    depthwise_conv2d, kARM, kInt8, kNCHW,
    paddle::lite::kernels::arm::ConvComputeInt8<PRECISION(kInt8)>, int8_out)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt8))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("Filter",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt8))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt8))})
    .Finalize();

REGISTER_LITE_KERNEL(
    depthwise_conv2d, kARM, kInt8, kNCHW,
    paddle::lite::kernels::arm::ConvComputeInt8<PRECISION(kFloat)>, fp32_out)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt8))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("Filter",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt8))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .Finalize();
