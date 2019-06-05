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

#include "paddle/fluid/lite/operators/concat_op.h"
#include "paddle/fluid/lite/core/op_lite.h"
#include "paddle/fluid/lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool ConcatOpLite::CheckShape() const {
  CHECK_GT_OR_FALSE(param_.x.size(), 1UL);
  CHECK_OR_FALSE(param_.output);
  return true;
}

bool ConcatOpLite::InferShape() const {
  std::vector<lite::DDim> input_dims;
  for (auto p : param_.x) {
    input_dims.push_back(p->dims());
  }
  size_t axis = static_cast<size_t>(param_.axis);
  const size_t n = input_dims.size();
  CHECK_GT_OR_FALSE(n, 0);
  auto &out_dims = input_dims[0];
  size_t in_zero_dims_size = out_dims.size();
  for (size_t i = 1; i < n; i++) {
    for (size_t j = 0; j < in_zero_dims_size; j++) {
      if (j == axis) {
        out_dims[axis] += input_dims[i][j];
      } else {
        CHECK_EQ_OR_FALSE(out_dims[j], input_dims[i][j]);
      }
    }
  }
  if (out_dims[axis] < 0) {
    out_dims[axis] = -1;
  }
  // Set output dims
  param_.output->Resize(lite::DDim(out_dims));
  return true;
}

// TODO(Superjomn) replace framework::OpDesc with a lite one.
bool ConcatOpLite::AttachImpl(const cpp::OpDesc &op_desc, lite::Scope *scope) {
  auto inputs = op_desc.Input("X");
  auto out = op_desc.Output("Out").front();

  for (auto var : inputs) {
    param_.x.push_back(scope->FindVar(var)->GetMutable<lite::Tensor>());
  }
  CHECK(scope->FindVar(out));
  param_.output = scope->FindVar(out)->GetMutable<lite::Tensor>();
  param_.axis = op_desc.GetAttr<int>("axis");

  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(concat, paddle::lite::operators::ConcatOpLite);
