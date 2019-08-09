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

#include "paddle/fluid/lite/operators/gru_op.h"
#include "paddle/fluid/lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool GruOp::CheckShape() const {
  CHECK_OR_FALSE(param_.x);
  CHECK_OR_FALSE(param_.weight);
  CHECK_OR_FALSE(param_.batchGate);
  CHECK_OR_FALSE(param_.batchResetHiddenPrev);
  CHECK_OR_FALSE(param_.batchHidden);
  CHECK_OR_FALSE(param_.hidden);

  auto weight_dims = param_.weight->dims();
  int frame_size = weight_dims[0];
  CHECK_EQ_OR_FALSE(weight_dims[1], frame_size * 3);
  if (param_.h0) {
    auto h0_dims = param_.h0->dims();
    CHECK_EQ_OR_FALSE(h0_dims[1], frame_size);
  }
  if (param_.bias) {
    auto bias_dims = param_.bias->dims();
    int bias_height = bias_dims[0];
    int bias_width = bias_dims[1];
    CHECK_EQ_OR_FALSE(bias_height, 1);
    CHECK_EQ_OR_FALSE(bias_width, frame_size * 3);
  }
  return true;
}

bool GruOp::InferShape() const {
  auto input_dims = param_.x->dims();
  auto weight_dims = param_.weight->dims();
  int frame_size = weight_dims[0];
  std::vector<int64_t> outShape{input_dims[0], frame_size};
  param_.batchGate->Resize(input_dims);
  param_.batchResetHiddenPrev->Resize(lite::DDim(outShape));
  param_.batchHidden->Resize(lite::DDim(outShape));
  param_.hidden->Resize(lite::DDim(outShape));
  param_.hidden->raw_tensor().set_lod(param_.x->lod());
  return true;
}

bool GruOp::AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) {
  param_.x =
      scope->FindVar(opdesc.Input("Input").front())->GetMutable<lite::Tensor>();
  param_.weight = scope->FindVar(opdesc.Input("Weight").front())
                      ->GetMutable<lite::Tensor>();

  param_.batchGate = scope->FindVar(opdesc.Output("BatchGate").front())
                         ->GetMutable<lite::Tensor>();
  param_.batchResetHiddenPrev =
      scope->FindVar(opdesc.Output("BatchResetHiddenPrev").front())
          ->GetMutable<lite::Tensor>();
  param_.batchHidden = scope->FindVar(opdesc.Output("BatchHidden").front())
                           ->GetMutable<lite::Tensor>();
  param_.hidden = scope->FindVar(opdesc.Output("Hidden").front())
                      ->GetMutable<lite::Tensor>();

  std::vector<std::string> input_arg_names = opdesc.InputArgumentNames();
  if (std::find(input_arg_names.begin(), input_arg_names.end(), "H0") !=
      input_arg_names.end()) {
    if (opdesc.Input("H0").size() != 0) {
      param_.h0 = scope->FindVar(opdesc.Input("H0").front())
                      ->GetMutable<lite::Tensor>();
    }
  }
  if (std::find(input_arg_names.begin(), input_arg_names.end(), "Bias") !=
      input_arg_names.end()) {
    param_.bias = scope->FindVar(opdesc.Input("Bias").front())
                      ->GetMutable<lite::Tensor>();
  }

  param_.activation = opdesc.GetAttr<std::string>("activation");
  param_.gate_activation = opdesc.GetAttr<std::string>("gate_activation");
  param_.is_reverse = opdesc.GetAttr<bool>("is_reverse");
  param_.origin_mode = opdesc.GetAttr<bool>("origin_mode");

  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(gru, paddle::lite::operators::GruOp);
