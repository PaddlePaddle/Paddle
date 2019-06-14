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
  // bias is optional.

  const auto in_dims = param_.x->dims();
  const auto filter_dims = param_.filter->dims();

  CHECK_OR_FALSE(in_dims.size() == 4 || in_dims.size() == 5);

  CHECK_EQ_OR_FALSE(in_dims.size(), filter_dims.size());
  CHECK_OR_FALSE(in_dims.size() - param_.strides.size() == 2U);
  CHECK_EQ_OR_FALSE(param_.paddings.size(), param_.strides.size());

  CHECK_EQ_OR_FALSE(in_dims[1], filter_dims[1] * param_.groups);
  CHECK_EQ_OR_FALSE(filter_dims[0] % param_.groups, 0);
  CHECK_EQ_OR_FALSE(filter_dims.size(), 4UL);

  return true;
}

inline int ConvOutputSize(int input_size, int filter_size, int dilation,
                          int padding, int stride) {
  const int dkernel = dilation * (filter_size - 1) + 1;
  int output_size = (input_size + 2 * padding - dkernel) / stride + 1;
  CHECK_GT_OR_FALSE(output_size, 0);

  return output_size;
}

bool ConvOpLite::InferShape() const {
  const auto in_dims = param_.x->dims();
  const auto filter_dims = param_.filter->dims();

  std::vector<int64_t> output_shape({in_dims[0], filter_dims[0]});
  for (size_t i = 0; i < param_.strides.size(); ++i) {
    output_shape.push_back(
        ConvOutputSize(in_dims[i + 2], filter_dims[i + 2], param_.dilations[i],
                       param_.paddings[i], param_.strides[i]));
  }

  // Set output dims
  param_.output->Resize(lite::DDim(output_shape));

  // share LoD
  // param_.output->set_lod(param_.x->lod());
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(conv2d, paddle::lite::operators::ConvOpLite);
REGISTER_LITE_OP(depthwise_conv2d, paddle::lite::operators::ConvOpLite);
