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

#include "paddle/fluid/lite/kernels/arm/batch_norm_compute.h"
#include "paddle/fluid/lite/arm/math/funcs.h"
#include "paddle/fluid/lite/core/op_registry.h"
#include "paddle/fluid/lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

void BatchNormCompute::PrepareForRun() {
  auto& param = this->Param<param_t>();
  auto x_dims = param.x->dims();
  bool global_stats = param.is_test || param.use_global_stats;
  if (global_stats) {
    int64_t channel_size = 0;
    switch (param.data_layout) {
      case DATALAYOUT(kNCHW):
        channel_size = x_dims[1];
        break;
      // case DATALAYOUT(kNHWC):
      //   channel_size = x_dims[x_dims.size() - 1];
      //   break;
      default:
        LOG(FATAL) << "Unknown storage order: "
                   << DataLayoutToStr(param.data_layout);
        break;
    }
    new_scale.Resize({channel_size});
    new_bias.Resize({channel_size});
    auto* scale_data = param.scale->mutable_data<float>();
    auto* bias_data = param.bias->mutable_data<float>();
    auto* mean_data = param.mean->mutable_data<float>();
    auto* variance_data = param.variance->mutable_data<float>();
    auto* new_scale_data = new_scale.mutable_data<float>();
    auto* new_bias_data = new_bias.mutable_data<float>();
    for (int c = 0; c < channel_size; c++) {
      float inv_scale = 1.f / (std::sqrt(variance_data[c] + param.epsilon));
      new_bias_data[c] =
          bias_data[c] - inv_scale * scale_data[c] * mean_data[c];
      new_scale_data[c] = inv_scale * scale_data[c];
    }
  }
}

void BatchNormCompute::Run() {
  auto& param = this->Param<param_t>();
  auto x_dims = param.x->dims();
  auto x_data = param.x->mutable_data<float>();
  auto y_data = param.y->mutable_data<float>();
  bool global_stats = param.is_test || param.use_global_stats;
  if (global_stats) {
    auto* new_scale_data = new_scale.mutable_data<float>();
    auto* new_bias_data = new_bias.mutable_data<float>();
    int64_t outer_size = 0;
    int64_t channel_size = 0;
    int64_t inner_size = 0;
    switch (param.data_layout) {
      case DATALAYOUT(kNCHW):
        outer_size = x_dims[0];
        channel_size = x_dims[1];
        inner_size = x_dims.Slice(2, x_dims.size()).production();
        lite::arm::math::scale(x_data, y_data, outer_size, channel_size,
                               inner_size, new_scale_data, new_bias_data);
        break;
      // case DATALAYOUT(kNHWC):
      //   outer_size = x_dims.Slice(0, x_dims.size() - 1).production();
      //   channel_size = x_dims[x_dims.size() - 1];
      //   lite::arm::math::scale(x_data, y_data, outer_size, channel_size,
      //                          new_scale_data, new_bias_data);
      //   break;
      default:
        LOG(FATAL) << "Unknown storage order: "
                   << DataLayoutToStr(param.data_layout);
        break;
    }
  } else {
    // TODO(hong19860320) calculate mean_out, variance_out, saved_mean and
    // saved_variance
  }
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(batch_norm, kARM, kFloat, kNCHW,
                     paddle::lite::kernels::arm::BatchNormCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Scale", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Mean", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Variance", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Y", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("MeanOut", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("VarianceOut", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("SavedMean", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("SavedVariance", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
