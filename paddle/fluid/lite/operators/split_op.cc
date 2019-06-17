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

#include "paddle/fluid/lite/operators/split_op.h"
#include "paddle/fluid/lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool SplitOp::CheckShape() const {
  CHECK_OR_FALSE(param_.x);
  CHECK_GT_OR_FALSE(param_.output.size(), 1UL);
  auto x_dims = param_.x->dims();
  auto x_rank = x_dims.size();
  CHECK_OR_FALSE(param_.axis >= -static_cast<int>(x_rank) &&
                 param_.axis < static_cast<int>(x_rank));
  return true;
}

bool SplitOp::InferShape() const {
  const auto &outs = param_.output;
  auto in_dims = param_.x->dims();
  int axis = param_.axis;
  int num = param_.num;
  const auto &sections = param_.sections;

  const int outs_number = outs.size();
  std::vector<lite::DDim> outs_dims;
  outs_dims.reserve(outs_number);

  if (num > 0) {
    int out_axis_dim = in_dims[axis] / num;
    for (int i = 0; i < outs_number; ++i) {
      auto dim = in_dims;
      dim[axis] = out_axis_dim;
      outs_dims.push_back(dim);
    }
  } else if (sections.size() > 0) {
    for (int i = 0; i < outs_number; ++i) {
      auto dim = in_dims;
      dim[axis] = sections[i];
      outs_dims.push_back(dim);
    }
  }

  for (int j = 0; j < outs_dims.size(); ++j) {
    outs[j]->Resize(outs_dims[j]);
  }

  return true;
}

bool SplitOp::AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) {
  param_.axis = opdesc.GetAttr<int>("axis");
  param_.num = opdesc.GetAttr<int>("num");
  param_.sections = opdesc.GetAttr<std::vector<int>>("sections");
  auto input = opdesc.Input("Input").front();
  auto outs = opdesc.Output("Out");
  param_.x = scope->FindVar(input)->GetMutable<lite::Tensor>();
  for (auto var : outs) {
    param_.output.push_back(scope->FindVar(var)->GetMutable<lite::Tensor>());
  }
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(split, paddle::lite::operators::SplitOp);
