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

#include "paddle/fluid/lite/operators/batch_norm_op.h"
#include "paddle/fluid/lite/core/op_registry.h"
namespace paddle {
namespace lite {
namespace operators {

bool BatchNormOp::CheckShape() const {
  CHECK_OR_FALSE(param_.x);
  CHECK_OR_FALSE(param_.bias);
  CHECK_OR_FALSE(param_.scale);
  CHECK_OR_FALSE(param_.mean);
  CHECK_OR_FALSE(param_.variance);
  CHECK_OR_FALSE(param_.y);
  if (!param_.is_test) {
    CHECK_OR_FALSE(param_.mean_out);
    CHECK_OR_FALSE(param_.variance_out);
    CHECK_OR_FALSE(param_.saved_mean);
    CHECK_OR_FALSE(param_.saved_variance);
  }
  auto x_dims = param_.x->dims();
  auto scale_dims = param_.scale->dims();
  auto bias_dims = param_.bias->dims();
  auto mean_dims = param_.mean->dims();
  auto variance_dims = param_.variance->dims();
  CHECK(x_dims.size() >= 2 && x_dims.size() <= 5)
      << "Input X must have 2 to 5 dimensions.";
  CHECK_EQ(scale_dims.size(), 1UL) << "Input Scale must have 1 dimensions.";
  CHECK_EQ(bias_dims.size(), 1UL) << "Input Bias must have 1 dimensions.";
  CHECK_EQ(mean_dims.size(), 1UL) << "Input Mean must have 1 dimensions.";
  CHECK_EQ(variance_dims.size(), 1UL)
      << "Input Variance must have 1 dimensions.";
  return true;
}

bool BatchNormOp::InferShape() const {
  auto x_dims = param_.x->dims();
  int64_t channel_size = 0;
  switch (param_.data_layout) {
    case DATALAYOUT(kNCHW):
      channel_size = x_dims[1];
      break;
    // case DATALAYOUT(kNHWC):
    //   channel_size = x_dims[x_dims.size() - 1];
    //   break;
    default:
      LOG(FATAL) << "Unknown storage order: "
                 << DataLayoutToStr(param_.data_layout);
      break;
  }
  if (!param_.is_test) {
    param_.mean_out->Resize({channel_size});
    param_.variance_out->Resize({channel_size});
    param_.saved_mean->Resize({channel_size});
    param_.saved_variance->Resize({channel_size});
  }
  param_.y->Resize(x_dims);
  return true;
}

bool BatchNormOp::AttachImpl(const cpp::OpDesc &op_desc, lite::Scope *scope) {
  param_.x = scope->FindVar(op_desc.Input("X").front())->GetMutable<Tensor>();
  param_.bias =
      scope->FindVar(op_desc.Input("Bias").front())->GetMutable<Tensor>();
  param_.scale =
      scope->FindVar(op_desc.Input("Scale").front())->GetMutable<Tensor>();
  param_.mean =
      scope->FindVar(op_desc.Input("Mean").front())->GetMutable<Tensor>();
  param_.variance =
      scope->FindVar(op_desc.Input("Variance").front())->GetMutable<Tensor>();
  param_.y = scope->FindVar(op_desc.Output("Y").front())->GetMutable<Tensor>();
  param_.is_test = op_desc.GetAttr<int>("is_test");
  param_.use_global_stats = op_desc.GetAttr<bool>("use_global_stats");
  if (!param_.is_test) {
    param_.mean_out =
        scope->FindVar(op_desc.Output("MeanOut").front())->GetMutable<Tensor>();
    param_.variance_out = scope->FindVar(op_desc.Output("VarianceOut").front())
                              ->GetMutable<Tensor>();
    param_.saved_mean = scope->FindVar(op_desc.Output("SavedMean").front())
                            ->GetMutable<Tensor>();
    param_.saved_variance =
        scope->FindVar(op_desc.Output("SavedVariance").front())
            ->GetMutable<Tensor>();
  }
  param_.epsilon = op_desc.GetAttr<float>("epsilon");
  param_.momentum = op_desc.GetAttr<float>("momentum");
  std::string data_layout = op_desc.GetAttr<std::string>("data_layout");
  CHECK_EQ(data_layout, "NCHW") << "TODO(hong19860320): Only support NCHW.";
  // param_.data_layout = StringToDataLayout(data_layout);
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(batch_norm, paddle::lite::operators::BatchNormOp);
