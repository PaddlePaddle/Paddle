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

#include "paddle/fluid/lite/operators/mul_op.h"
#include "paddle/fluid/lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool MulOpLite::CheckShape() const {
  CHECK_OR_FALSE(param_.x);
  CHECK_OR_FALSE(param_.y);
  CHECK_OR_FALSE(param_.output);
  // bias is optional.

  const auto x_dims = param_.x->dims();
  const auto y_dims = param_.y->dims();

  CHECK_GT_OR_FALSE(x_dims.size(), static_cast<size_t>(param_.x_num_col_dims));
  CHECK_GT_OR_FALSE(y_dims.size(), static_cast<size_t>(param_.y_num_col_dims));

#ifndef LITE_WITH_LIGHT_WEIGHT_FRAMEWORK
  auto x_mat_dims =
      framework::flatten_to_2d(x_dims.data(), param_.x_num_col_dims);
  auto y_mat_dims =
      framework::flatten_to_2d(y_dims.data(), param_.y_num_col_dims);

  PADDLE_ENFORCE_EQ(x_mat_dims[1], y_mat_dims[0],
                    "First matrix's width must be equal with second matrix's"
                    "height. %s, %s",
                    x_mat_dims[1], y_mat_dims[0]);
#endif

  return true;
}

bool MulOpLite::InferShape() const {
  const auto x_dims = param_.x->dims();
  const auto y_dims = param_.y->dims();

  // Set output dims
  std::vector<int64_t> out_dims(
      param_.x_num_col_dims + y_dims.size() - param_.y_num_col_dims, 0);
  for (int i = 0; i < param_.x_num_col_dims; ++i) {
    out_dims[i] = x_dims[i];
  }

  for (auto i = static_cast<size_t>(param_.y_num_col_dims); i < y_dims.size();
       ++i) {
    out_dims[i] = y_dims[i];
  }

  param_.output->Resize(lite::DDim(out_dims));

  // share LoD
  // param_.output->set_lod(param_.input->lod());
  return true;
}

#ifdef LITE_WITH_X86

bool MulGradOpLite::CheckShape() const {
  CHECK_OR_FALSE(param_.x);
  CHECK_OR_FALSE(param_.y);
  CHECK_OR_FALSE(param_.output_grad);

  return true;
}

bool MulGradOpLite::InferShape() const {
  if (param_.x_grad) param_.x_grad->Resize(param_.x->dims());
  if (param_.y_grad) param_.y_grad->Resize(param_.y->dims());
  return true;
}

bool MulGradOpLite::AttachImpl(const cpp::OpDesc &op_desc, lite::Scope *scope) {
  auto X_name = op_desc.Input("X").front();
  auto Y_name = op_desc.Input("Y").front();
  auto Out_grad_name = op_desc.Input(framework::GradVarName("Out")).front();

  if (op_desc.Output(framework::GradVarName("X")).size()) {
    auto X_grad_name = op_desc.Output(framework::GradVarName("X")).front();
    param_.x_grad = GetMutableVar<lite::Tensor>(scope, X_grad_name);
  }

  if (op_desc.Output(framework::GradVarName("Y")).size()) {
    auto Y_grad_name = op_desc.Output(framework::GradVarName("Y")).front();
    param_.y_grad = GetMutableVar<lite::Tensor>(scope, Y_grad_name);
  }

  param_.x = GetVar<lite::Tensor>(scope, X_name);
  param_.y = GetVar<lite::Tensor>(scope, Y_name);
  param_.output_grad = GetVar<lite::Tensor>(scope, Out_grad_name);

  return true;
}
#endif

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(mul, paddle::lite::operators::MulOpLite);
#ifdef LITE_WITH_X86
REGISTER_LITE_OP(mul_grad, paddle::lite::operators::MulGradOpLite);
#endif
