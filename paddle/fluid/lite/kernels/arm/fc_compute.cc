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
#include "paddle/fluid/lite/arm/math/funcs.h"
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
    float* packed_in = static_cast<float*>(ctx.workspace_data<float>()) +
                       ctx.l2_cache_size() / sizeof(float);
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
