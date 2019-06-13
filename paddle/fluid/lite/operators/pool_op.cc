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

#include "paddle/fluid/lite/operators/pool_op.h"
#include "paddle/fluid/lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool PoolOpLite::CheckShape() const {
  CHECK_OR_FALSE(param_.x);
  CHECK_OR_FALSE(param_.output);

  const auto& x_dims = param_.x->dims();
  const auto& ksize = param_.ksize;
  const auto& strides = param_.strides;
  const auto& paddings = param_.paddings;

  // "Pooling intput should be 4-D or 5-D tensor."
  CHECK_OR_FALSE(x_dims.size() == 4 || x_dims.size() == 5);
  // Input size and pooling size should be consistent.
  CHECK_OR_FALSE(x_dims.size() - ksize.size() == 2U);
  // Strides size and pooling size should be the same.
  CHECK_OR_FALSE(ksize.size() == strides.size());
  // Paddings size and pooling size should be the same.
  CHECK_OR_FALSE(ksize.size() == paddings.size());

  return true;
}

int PoolOutputSize(int input_size, int filter_size, int padding, int stride,
                   bool ceil_mode) {
  int output_size;
  if (!ceil_mode) {
    output_size = (input_size - filter_size + 2 * padding) / stride + 1;
  } else {
    output_size =
        (input_size - filter_size + 2 * padding + stride - 1) / stride + 1;
  }
  return output_size;
}

bool PoolOpLite::InferShape() const {
  const auto x_dims = param_.x->dims();
  std::vector<int>& ksize = param_.ksize;
  if (param_.global_pooling) {
    ksize.resize(static_cast<size_t>(x_dims.size()) - 2);
    for (size_t i = 0; i < ksize.size(); ++i) {
      param_.paddings[i] = 0;
      ksize[i] = static_cast<int>(x_dims[i + 2]);
    }
  }

  std::vector<int64_t> output_shape({x_dims[0], x_dims[1]});
  if (param_.adaptive) {
    output_shape.insert(output_shape.end(), param_.ksize.begin(),
                        param_.ksize.end());
  } else {
    for (size_t i = 0; i < param_.ksize.size(); ++i) {
      output_shape.push_back(
          PoolOutputSize(x_dims[i + 2], param_.ksize[i], param_.paddings[i],
                         param_.strides[i], param_.ceil_mode));
    }
  }
  param_.output->Resize(lite::DDim(output_shape));

  // ctx->SetOutputDim("Out", framework::make_ddim(output_shape));
  // ctx->ShareLoD("X", "Out");
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(pool2d, paddle::lite::operators::PoolOpLite);
