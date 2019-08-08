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
#include <vector>
#include "paddle/fluid/lite/api/paddle_place.h"
#include "paddle/fluid/lite/arm/math/funcs.h"
#include "paddle/fluid/lite/arm/math/gemm_prepacked_int8.h"
#include "paddle/fluid/lite/arm/math/gemv_arm_int8.h"
#include "paddle/fluid/lite/core/op_registry.h"
#include "paddle/fluid/lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

void FcCompute::PrepareForRun() {
  auto& param = this->Param<operators::FcParam>();
  auto x_dims = param.input->dims();
  auto w_dims = param.w->dims();

  auto& ctx = this->ctx_->template As<ARMContext>();

  CHECK_GE(x_dims.size(), 2UL);
  CHECK_EQ(w_dims.size(), 2UL);
  CHECK_EQ(param.output->dims().size(), 2UL);

  m_ = x_dims.Slice(0, param.in_num_col_dims).production();
  k_ = x_dims.Slice(param.in_num_col_dims, x_dims.size()).production();
  n_ = w_dims[1];
  CHECK_EQ(k_, static_cast<int>(w_dims[0]));

  if (m_ == 1) {
    if (!transed_weight_) {
      transed_weight_ = new Tensor;
    }
    transed_weight_->Resize({n_, k_});
    const auto* w_data = param.w->data<float>();
    auto* t_data = transed_weight_->mutable_data<float>();
    int i = 0;

    for (int nn = 0; nn < n_; ++nn) {
      for (int kk = 0; kk < k_; ++kk) {
        t_data[i++] = w_data[kk * n_ + nn];
      }
    }
  }

  if (m_ > 1) {
    int hblock = lite::arm::math::get_hblock(ctx.arch());
    int m_round = hblock * ((m_ + hblock - 1) / hblock);
    ctx.ExtendWorkspace(DDimLite(std::vector<int64_t>({m_round * k_})));
  }
}

void FcCompute::Run() {
  auto& param = this->Param<operators::FcParam>();

  const auto* i_data = param.input->data<float>();
  const auto* w_data = param.w->data<float>();
  const auto* b_data = param.bias ? param.bias->data<float>() : nullptr;
  auto* o_data = param.output->mutable_data<float>();

  auto& ctx = this->ctx_->template As<ARMContext>();
  if (m_ > 1) {
    float* packed_in =
        ctx.workspace_data<float>() + ctx.l2_cache_size() / sizeof(float);
    lite::arm::math::prepackA(packed_in, i_data, k_, 0, m_, 0, k_, false, &ctx);
    lite::arm::math::sgemm_prepack(packed_in, w_data, b_data, o_data, m_, n_,
                                   k_, false, false, false, &ctx);
    if (param.bias) {
      CHECK_EQ(param.bias->numel(), n_);
      lite::arm::math::fill_bias_fc(o_data, b_data, m_, n_);
    }
  } else {
    CHECK(transed_weight_);
    const auto* t_data = transed_weight_->data<float>();

    lite::arm::math::sgemv(t_data, i_data, o_data, false, n_, k_,
                           b_data != nullptr, b_data, false);
  }
}

template <PrecisionType Ptype_out>
void FcComputeInt8<Ptype_out>::PrepareForRun() {
  auto& param = this->Param<operators::FcParam>();
  auto x_dims = param.input->dims();
  auto w_dims = param.w->dims();

  auto& ctx = this->ctx_->template As<ARMContext>();
  if (!tmp_int32_out_) {
    tmp_int32_out_ = new Tensor;
    tmp_int32_out_->Resize(param.output->dims());
  }

  CHECK_GE(x_dims.size(), 2UL);
  CHECK_EQ(w_dims.size(), 2UL);
  CHECK_EQ(param.output->dims().size(), 2UL);

  this->m_ = x_dims.Slice(0, param.in_num_col_dims).production();
  this->k_ = x_dims.Slice(param.in_num_col_dims, x_dims.size()).production();
  this->n_ = w_dims[1];
  CHECK_EQ(k_, static_cast<int>(w_dims[0]));

  if (this->m_ == 1) {
    if (!this->transed_weight_) {
      this->transed_weight_ = new Tensor;
    }
    this->transed_weight_->Resize({this->n_, this->k_});
    const auto* w_data = param.w->template data<int8_t>();
    auto* t_data = this->transed_weight_->template mutable_data<int8_t>();
    int i = 0;

    for (int nn = 0; nn < this->n_; ++nn) {
      for (int kk = 0; kk < this->k_; ++kk) {
        t_data[i++] = w_data[kk * this->n_ + nn];
      }
    }
  }

  if (this->m_ > 1) {
    int hblock = lite::arm::math::get_hblock(ctx.arch());
    int m_round = hblock * ((this->m_ + hblock - 1) / hblock);
    ctx.ExtendWorkspace(DDimLite(std::vector<int64_t>({m_round * this->k_})));
  }
}

template <PrecisionType Ptype_out>
void FcComputeInt8<Ptype_out>::Run() {
  auto& param = this->Param<operators::FcParam>();

  const auto* i_data = param.input->template data<int8_t>();
  const auto* w_data = param.w->template data<int8_t>();
  const auto* b_data = param.bias ? param.bias->template data<int>() : nullptr;
  int* o_data = nullptr;

  auto& ctx = this->ctx_->template As<ARMContext>();

  o_data = this->tmp_int32_out_->template mutable_data<int>();
  if (m_ > 1) {
    int8_t* packed_in =
        static_cast<int8_t*>(ctx.template workspace_data<int8_t>()) +
        ctx.l2_cache_size() / sizeof(int8_t);
    lite::arm::math::prepackA_int8(packed_in, i_data, k_, 0, m_, 0, k_, false);
    lite::arm::math::gemm_prepack_int8(packed_in, w_data, b_data, o_data, m_,
                                       n_, k_, false, false, false, nullptr,
                                       &ctx);
    if (param.bias) {
      CHECK_EQ(param.bias->numel(), n_);
      lite::arm::math::fill_bias_fc(o_data, b_data, m_, n_);
    }
  } else {
    CHECK(transed_weight_);
    const auto* t_data = transed_weight_->template data<int8_t>();
    lite::arm::math::gemv_int8(t_data, i_data, o_data, false, n_, k_, nullptr,
                               b_data != nullptr, b_data, false);
  }

  float i_scale = param.input_scale;
  std::vector<float> weight_scale = param.weight_scale;
  if (Ptype_out == PRECISION(kInt8)) {
    float o_scale = param.output_scale;
    param.output->template mutable_data<int8_t>();
    lite::arm::math::trans_tensor_dtype<PRECISION(kInt32), PRECISION(kInt8)>(
        tmp_int32_out_, param.output, i_scale, o_scale, weight_scale);
  } else if (Ptype_out == PRECISION(kFloat)) {
    param.output->template mutable_data<float>();
    lite::arm::math::trans_tensor_dtype<PRECISION(kInt32), PRECISION(kFloat)>(
        tmp_int32_out_, param.output, i_scale, 1.f, weight_scale);
  } else {
    LOG(ERROR) << "unsupported precision type!!";
  }
}

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

REGISTER_LITE_KERNEL(
    fc, kARM, kInt8, kNCHW,
    paddle::lite::kernels::arm::FcComputeInt8<PRECISION(kInt8)>, int8out)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt8))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("W", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt8))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt8))})
    .Finalize();

REGISTER_LITE_KERNEL(
    fc, kARM, kInt8, kNCHW,
    paddle::lite::kernels::arm::FcComputeInt8<PRECISION(kFloat)>, fp32out)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt8))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("W", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt8))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .Finalize();
