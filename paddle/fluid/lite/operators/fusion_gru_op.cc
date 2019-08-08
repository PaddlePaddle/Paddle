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

#include "paddle/fluid/lite/operators/fusion_gru_op.h"
#include "paddle/fluid/lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool FusionGruOp::CheckShape() const {
  CHECK_OR_FALSE(param_.x);
  CHECK_OR_FALSE(param_.weightH);
  CHECK_OR_FALSE(param_.weightX);
  CHECK_OR_FALSE(param_.xx);
  CHECK_OR_FALSE(param_.hidden);
  auto x_dims = param_.x->dims();
  CHECK_EQ_OR_FALSE(x_dims.size(), 2);
  auto wx_dims = param_.weightX->dims();
  CHECK_EQ_OR_FALSE(wx_dims.size(), 2);
  CHECK_EQ_OR_FALSE(wx_dims[0], x_dims[1]);

  int frame_size = wx_dims[1] / 3;
  auto wh_dims = param_.weightH->dims();
  CHECK_EQ_OR_FALSE(wh_dims.size(), 2);
  CHECK_EQ_OR_FALSE(wh_dims[0], frame_size);
  CHECK_EQ_OR_FALSE(wh_dims[1], 3 * frame_size);

  if (param_.h0) {
    auto h0_dims = param_.h0;
    CHECK_EQ_OR_FALSE(h0_dims[1], frame_size);
  }

  if (param_.bias) {
    auto b_dims = param_.bias->dims();
    CHECK_EQ_OR_FALSE(b_dims.size(), 2);
    CHECK_EQ_OR_FALSE(b_dims[0], 1);
    CHECK_EQ_OR_FALSE(b_dims[1], frame_size * 3);
  }

  return true;
}

bool FusionGruOp::InferShape() const {
  auto x_dims = param_.x->dims();
  auto wx_dims = param_.weightX->dims();
  int frame_size = wx_dims[1] / 3;
  std::vector<int64_t> out_shape{x_dims[0], frame_size};
  param_.hidden->Resize(lite::DDim(out_shape));
  param_.hidden->raw_tensor().set_lod(param_.x->lod());

  int xx_width;
  if (param_.use_seq) {
    xx_width = wx_dims[1];
  } else {
    xx_width = x_dims[1] > wx_dims[1] ? wx_dims[1] : x_dims[1];
    CHECK_OR_FALSE(param_.reorderedH0);
    CHECK_OR_FALSE(param_.batchedInput);
    CHECK_OR_FALSE(param_.batchedOut);
    std::vector<int64_t> batched_input_shape{x_dims[0], wx_dims[1]};
    param_.batchedInput->Resize(lite::DDim(batched_input_shape);
    param_.batchedOut->Resize(lite::DDim(out_shape));
  }
  std::vector<int64_t> xx_shape{x_dins[0], xx_width};
  param_.xx->Resize(lite::DDim(xx_shape));
  param_.xx->raw_tensor.set_lod(param.x->lod());

  return true;
}

bool FusionGruOp::AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) {
  param_.x = scope->FindVar(opdesc.Input("X").front())->GetMutable<lite::Tensor>());
  if (std::find(input_arg_names.begin(), input_arg_names.end(), "H0") !=
      input_arg_names.end()) {
    if (opdesc.Input("H0").size() != 0) {
      param_.h0 = scope->FindVar(opdesc.Input("H0").front())
                      ->GetMutable<lite::Tensor>();
    }
  }
  param_.weightX = scope->FindVar(opdesc.Input("WeightX").front())->GetMutable<lite::Tensor>());
  param_.weightH = scope->FindVar(opdesc.Input("WeightH").front())->GetMutable<lite::Tensor>());

  if (std::find(input_arg_names.begin(), input_arg_names.end(), "Bias") !=
      input_arg_names.end()) {
    param_.bias = scope->FindVar(opdesc.Input("Bias").front())
                      ->GetMutable<lite::Tensor>();
  }

  param_.reorderedH0 = scope->FindVar(opdesc.Input("ReorderedH0").front())->GetMutable<lite::Tensor>());
  param_.xx = scope->FindVar(opdesc.Input("XX").front())->GetMutable<lite::Tensor>());
  param_.batchedInput = scope->FindVar(opdesc.Input("BatchedInput").front())->GetMutable<lite::Tensor>());
  param_.batchedOut = scope->FindVar(opdesc.Input("BatchedOut").front())->GetMutable<lite::Tensor>());
  param_.hidden= scope->FindVar(opdesc.Input("Hidden").front())->GetMutable<lite::Tensor>());

  param_.activation = opdesc.GetAttr<std::string>("activation");
  param_.gate_activation = opdesc.GetAttr<std::string>("gate_activation");
  param_.is_reverse = opdesc.GetAttr<bool>("is_reverse");
  param_.use_seq = opdesc.GetAttr<bool>("use_seq");
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(fusion_gru, paddle::lite::operators::FusionGruOp);
