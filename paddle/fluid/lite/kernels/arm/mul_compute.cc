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

#include "paddle/fluid/lite/kernels/arm/mul_compute.h"
#include <vector>
#include "paddle/fluid/lite/arm/math/funcs.h"
#include "paddle/fluid/lite/core/op_registry.h"
#include "paddle/fluid/lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

void MulCompute::PrepareForRun() {
  auto& ctx = this->ctx_->template As<ARMContext>();
}

void MulCompute::Run() {
  auto& param = Param<param_t>();

  const auto* x_data = param.x->data<float>();
  const auto* y_data = param.y->data<float>();
  auto* o_data = param.output->mutable_data<float>();

  m_ = static_cast<int>(
      param.x->dims().Slice(0, param.x_num_col_dims).production());
  int x_w =
      static_cast<int>(param.x->dims()
                           .Slice(param.x_num_col_dims, param.x->dims().size())
                           .production());
  int y_h = static_cast<int>(
      param.y->dims().Slice(0, param.y_num_col_dims).production());
  n_ = static_cast<int>(param.y->dims()
                            .Slice(param.y_num_col_dims, param.y->dims().size())
                            .production());

  CHECK_EQ(x_w, y_h) << "x_w must be equal with y_h";
  k_ = x_w;

  if (n_ == 1) {
    lite::arm::math::sgemv(x_data, y_data, o_data, false, m_, k_, false,
                           nullptr, false);

  } else {
    constexpr bool is_tranposed_y = false;
    auto& ctx = this->ctx_->template As<ARMContext>();
    int hblock = lite::arm::math::get_hblock(ctx.arch());
    int m_round = hblock * ((m_ + hblock - 1) / hblock);
    ctx.ExtendWorkspace(DDimLite(std::vector<int64_t>({m_round * k_})));

    float* packed_x = static_cast<float*>(ctx.workspace_data<float>()) +
                      ctx.l2_cache_size() / sizeof(float);
    lite::arm::math::prepackA(packed_x, x_data, k_, 0, m_, 0, k_, false, &ctx);
    lite::arm::math::sgemm_prepack(packed_x, y_data, nullptr, o_data, m_, n_,
                                   k_, false, false, is_tranposed_y, &ctx);
  }
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(mul, kARM, kFloat, kNCHW,
                     paddle::lite::kernels::arm::MulCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
