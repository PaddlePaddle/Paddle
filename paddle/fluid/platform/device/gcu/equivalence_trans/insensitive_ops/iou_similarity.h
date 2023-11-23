/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/platform/device/gcu/register/register.h"

namespace paddle {
namespace platform {
namespace gcu {
const char *const kIOUSimilarity = "iou_similarity";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(gcu_builder,
                               node,
                               map_inputs,
                               running_mode,
                               IOUSimilarityEquivalenceTrans) {
  // auto *op = node->Op();
  // bool normalized = PADDLE_GET_CONST(bool, op->GetAttr("box_normalized"));
  GcuOp x = *(map_inputs["X"].at(0));
  GcuOp y = *(map_inputs["Y"].at(0));
  auto x_dtype = x.GetType().GetPrimitiveType();
  int64_t N = x.GetType().GetShape()[0];
  int64_t M = y.GetType().GetShape()[0];
  // auto x_t = builder::Transpose(x, {1, 0});
  // auto y_t = builder::Transpose(y, {1, 0});
  auto xmin1 = builder::SliceInDim(x, 0, 1, 1, 1);
  auto ymin1 = builder::SliceInDim(x, 1, 2, 1, 1);
  auto xmax1 = builder::SliceInDim(x, 2, 3, 1, 1);
  auto ymax1 = builder::SliceInDim(x, 3, 4, 1, 1);
  auto xmin2 = builder::SliceInDim(y, 0, 1, 1, 1);
  auto ymin2 = builder::SliceInDim(y, 1, 2, 1, 1);
  auto xmax2 = builder::SliceInDim(y, 2, 3, 1, 1);
  auto ymax2 = builder::SliceInDim(y, 3, 4, 1, 1);

  xmin1 = builder::Reshape(xmin1, {{N, 1}, x_dtype});
  ymin1 = builder::Reshape(ymin1, {{N, 1}, x_dtype});
  xmax1 = builder::Reshape(xmax1, {{N, 1}, x_dtype});
  ymax1 = builder::Reshape(ymax1, {{N, 1}, x_dtype});
  xmin2 = builder::Reshape(xmin2, {{1, M}, x_dtype});
  ymin2 = builder::Reshape(ymin2, {{1, M}, x_dtype});
  xmax2 = builder::Reshape(xmax2, {{1, M}, x_dtype});
  ymax2 = builder::Reshape(ymax2, {{1, M}, x_dtype});

  auto w1 = xmax1 - xmin1;
  auto h1 = ymax1 - ymin1;
  auto w2 = xmax2 - xmin2;
  auto h2 = ymax2 - ymin2;
  //   if (!normalized) {
  //   }
  auto area1 = w1 * h1;
  auto area2 = w2 * h2;
  auto inter_xmax = builder::Min(xmax1, xmax2);
  auto inter_ymax = builder::Min(ymax1, ymax2);
  auto inter_xmin = builder::Max(xmin1, xmin2);
  auto inter_ymin = builder::Max(ymin1, ymin2);
  auto inter_w = inter_xmax - inter_xmin;
  auto inter_h = inter_ymax - inter_ymin;
  //   if (!normalized) {
  //   }
  // Replace Max(inter_x, 0) with Relu(inter_x), to improve performance.
  float one_data = 1.;
  auto one_op = builder::Const(
      gcu_builder, static_cast<void *>(&one_data), builder::Type(x_dtype));
  float zero_data = 0.;
  auto zero_op = builder::Const(
      gcu_builder, static_cast<void *>(&zero_data), builder::Type(x_dtype));
  inter_w = builder::Relu(inter_w);
  inter_h = builder::Relu(inter_h);
  auto out = inter_w * inter_h;
  auto union_area = area1 + area2 - out;
  auto pred = builder::Equal(union_area, zero_op);
  auto new_union = builder::Select(pred, one_op, union_area);
  auto result = out / new_union;
  return std::make_shared<GcuOp>(result);
}

EQUIVALENCE_TRANS_FUNC_REG(kIOUSimilarity,
                           INSENSITIVE,
                           IOUSimilarityEquivalenceTrans);
}  // namespace gcu
}  // namespace platform
}  // namespace paddle
