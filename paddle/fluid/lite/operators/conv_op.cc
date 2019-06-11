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

#include "paddle/fluid/lite/operators/conv_op.h"
#include <vector>
#include "paddle/fluid/lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool ConvOpLite::CheckShape() const {
  CHECK_OR_FALSE(param_.x);
  CHECK_OR_FALSE(param_.output);
  CHECK_OR_FALSE(param_.filter);
  return true;
}

bool ConvOpLite::InferShape() const {
  auto in_dims = param_.x->dims();
  auto filter_dims = param_.filter->dims();
  std::vector<int> strides = param_.strides;
  std::vector<int> paddings = param_.paddings;
  int groups = param_.groups;
  std::vector<int> dilations = param_.dilations;

  CHECK_OR_FALSE(in_dims.size() == 4 || in_dims.size() == 5);
  CHECK_EQ_OR_FALSE(in_dims.size(), filter_dims.size());
  CHECK_OR_FALSE(in_dims.size() - strides.size() == 2U);
  CHECK_EQ_OR_FALSE(paddings.size(), strides.size());
  CHECK_EQ_OR_FALSE(in_dims[1], filter_dims[1] * groups);
  CHECK_EQ_OR_FALSE(filter_dims[0] % groups, 0);

  std::vector<int64_t> output_shape({in_dims[0], filter_dims[0]});
  for (size_t i = 0; i < strides.size(); ++i) {
    output_shape.push_back(ConvOutputSize(in_dims[i + 2], filter_dims[i + 2],
                                          dilations[i], paddings[i],
                                          strides[i]));
  }
  param_.output->Resize(lite::DDim(output_shape));
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(conv2d, paddle::lite::operators::ConvOpLite);
REGISTER_LITE_OP(depthwise_conv2d, paddle::lite::operators::ConvOpLite);
