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
const char *const kPool2D = "pool2d";
const char *const kPool2DGrad = "pool2d_grad";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, Pool2DEquivalenceTrans) {
  auto *op = node->Op();
  auto adaptive = false;
  if (op->HasAttr("adaptive")) {
    adaptive = PADDLE_GET_CONST(bool, op->GetAttr("adaptive"));
  }
  auto global_pooling = PADDLE_GET_CONST(bool, op->GetAttr("global_pooling"));
  auto padding_algorithm =
      PADDLE_GET_CONST(std::string, op->GetAttr("padding_algorithm"));
  auto pooling_type =
      PADDLE_GET_CONST(std::string, op->GetAttr("pooling_type"));
  auto data_format = PADDLE_GET_CONST(std::string, op->GetAttr("data_format"));
  auto ksize = PADDLE_GET_CONST(std::vector<int>, op->GetAttr("ksize"));
  auto strides_i32 = PADDLE_GET_CONST(std::vector<int>, op->GetAttr("strides"));
  auto paddings_i32 =
      PADDLE_GET_CONST(std::vector<int>, op->GetAttr("paddings"));
  auto ceil_mode = PADDLE_GET_CONST(bool, op->GetAttr("ceil_mode"));
  // auto exclusive = PADDLE_GET_CONST(bool, op->GetAttr("exclusive"));
  std::vector<int64_t> strides;
  for (auto dim : strides_i32) {
    strides.emplace_back(static_cast<int64_t>(dim));
  }
  std::vector<int64_t> padding;
  for (auto dim : paddings_i32) {
    padding.emplace_back(static_cast<int64_t>(dim));
  }
  auto input = *(map_inputs["X"].at(0));
  std::vector<int64_t> spatial_dims;
  auto input_shape = input.GetType().GetShape();
  int64_t ih = 0;
  int64_t iw = 0;
  if (data_format == "NCHW") {
    spatial_dims = {2, 3};
    ih = input_shape[2];
    iw = input_shape[3];
  } else if (data_format == "NHWC") {
    spatial_dims = {1, 2};
    ih = input_shape[1];
    iw = input_shape[2];
  }
  int64_t kh = 0;
  int64_t kw = 0;
  if (adaptive) {
    std::vector<int64_t> output_spatial_size;
    for (auto dim : ksize) {
      output_spatial_size.emplace_back(static_cast<int64_t>(dim));
    }
    if (ih % output_spatial_size[0] != 0 || iw % output_spatial_size[1] != 0) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "only support: MOD(oh, ih) == 0 && MOD(ow, iw) == 0"));
      return nullptr;
    }
    kh = ih / output_spatial_size[0];
    kw = iw / output_spatial_size[1];
    strides = {kh, kw};
  } else {
    kh = static_cast<int64_t>(ksize[0]);
    kw = static_cast<int64_t>(ksize[1]);
  }
  if (padding_algorithm == "SAME") {
    auto pad_h = get_same_padding_value(ih, kh, strides[0]);
    auto pad_w = get_same_padding_value(iw, kw, strides[1]);
    padding = {pad_h[0], pad_h[1], pad_w[0], pad_w[1]};
  } else {
    if (padding.size() == 1) {
      padding = {padding[0], padding[0], padding[0], padding[0]};
    } else if (padding.size() == 2) {
      padding = {padding[0], padding[0], padding[1], padding[1]};
    } else if (padding.size() == 8) {
      if (data_format == "NCHW") {
        padding = {padding[4], padding[5], padding[6], padding[7]};
      } else if (data_format == "NHWC") {
        padding = {padding[2], padding[3], padding[4], padding[5]};
      }
    }
  }
  std::vector<int64_t> kernel_shape = {kh, kw};
  bool do_transpose =
      input.GetType().GetShape().size() == 4 && data_format == "NCHW";
  do_transpose = do_transpose && (running_mode == RunningMode::ADAPTIVE);
  if (do_transpose) {
    if (pooling_type == "max") {
      if (global_pooling) {
        input = builder::Transpose(input, {0, 2, 3, 1});
        spatial_dims = {1, 2};
        auto tmp_op = std::make_shared<GcuOp>(
            builder::GlobalMaxPool(input, spatial_dims));
        return std::make_shared<GcuOp>(
            builder::Transpose(*tmp_op, {0, 3, 1, 2}));
      }
      input = builder::Transpose(input, {0, 2, 3, 1});
      auto tmp_op = std::make_shared<GcuOp>(builder::MaxPool2D(input,
                                                               kernel_shape,
                                                               ceil_mode,
                                                               false,
                                                               "NOTSET",
                                                               "NHWC",
                                                               strides,
                                                               padding));
      return std::make_shared<GcuOp>(builder::Transpose(*tmp_op, {0, 3, 1, 2}));
    } else if (pooling_type == "avg") {
      if (global_pooling) {
        input = builder::Transpose(input, {0, 2, 3, 1});
        spatial_dims = {1, 2};
        std::vector<int64_t> out_shape = {input.GetType().GetShape().at(0),
                                          1,
                                          1,
                                          input.GetType().GetShape().at(3)};
        // force output shape because of hlir infershape question
        auto out_type =
            builder::Type(out_shape, input.GetType().GetPrimitiveType());
        auto tmp_op = std::make_shared<GcuOp>(
            builder::GlobalAveragePool(input, spatial_dims, out_type));
        return std::make_shared<GcuOp>(
            builder::Transpose(*tmp_op, {0, 3, 1, 2}));
      }
      input = builder::Transpose(input, {0, 2, 3, 1});
      spatial_dims = {1, 2};
      auto tmp_op = std::make_shared<GcuOp>(builder::AveragePool(input,
                                                                 spatial_dims,
                                                                 kernel_shape,
                                                                 ceil_mode,
                                                                 false,
                                                                 strides,
                                                                 padding,
                                                                 "NOTSET"));
      return std::make_shared<GcuOp>(builder::Transpose(*tmp_op, {0, 3, 1, 2}));
    } else {
      PADDLE_THROW(
          platform::errors::Unimplemented("Unsupported "
                                          "pooling_type: %s",
                                          pooling_type.c_str()));
      return nullptr;
    }
  } else {
    if (pooling_type == "max") {
      if (global_pooling) {
        return std::make_shared<GcuOp>(
            builder::GlobalMaxPool(input, spatial_dims));
      }
      return std::make_shared<GcuOp>(builder::MaxPool2D(input,
                                                        kernel_shape,
                                                        ceil_mode,
                                                        false,
                                                        "NOTSET",
                                                        data_format.c_str(),
                                                        strides,
                                                        padding));
    } else if (pooling_type == "avg") {
      if (global_pooling) {
        return std::make_shared<GcuOp>(
            builder::GlobalAveragePool(input, spatial_dims));
      }
      return std::make_shared<GcuOp>(builder::AveragePool(input,
                                                          spatial_dims,
                                                          kernel_shape,
                                                          ceil_mode,
                                                          false,
                                                          strides,
                                                          padding,
                                                          "NOTSET"));
    } else {
      PADDLE_THROW(
          platform::errors::Unimplemented("Unsupported pooling"
                                          "_type: %s",
                                          pooling_type.c_str()));
      return nullptr;
    }
  }
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, Pool2DGradEquivalenceTrans) {
  auto *op = node->Op();
  builder::Op out_grad = *(map_inputs["Out@GRAD"].at(0));
  builder::Op in = *(map_inputs["X"].at(0));
  auto adaptive = false;
  if (op->HasAttr("adaptive")) {
    adaptive = PADDLE_GET_CONST(bool, op->GetAttr("adaptive"));
  }
  auto global_pooling = PADDLE_GET_CONST(bool, op->GetAttr("global_pooling"));
  auto padding_algorithm =
      PADDLE_GET_CONST(std::string, op->GetAttr("padding_algorithm"));
  auto pooling_type =
      PADDLE_GET_CONST(std::string, op->GetAttr("pooling_type"));
  auto data_format = PADDLE_GET_CONST(std::string, op->GetAttr("data_format"));
  auto ksize_i32 = PADDLE_GET_CONST(std::vector<int>, op->GetAttr("ksize"));
  auto strides_i32 = PADDLE_GET_CONST(std::vector<int>, op->GetAttr("strides"));
  auto paddings_i32 =
      PADDLE_GET_CONST(std::vector<int>, op->GetAttr("paddings"));
  std::vector<int64_t> ksize;
  for (auto dim : ksize_i32) {
    ksize.emplace_back(static_cast<int64_t>(dim));
  }
  std::vector<int64_t> strides;
  for (auto dim : strides_i32) {
    strides.emplace_back(static_cast<int64_t>(dim));
  }
  std::vector<int64_t> padding;
  for (auto dim : paddings_i32) {
    padding.emplace_back(static_cast<int64_t>(dim));
  }
  std::vector<int64_t> spatial_dims;
  auto input_shape = in.GetType().GetShape();
  auto ptype = in.GetType().GetPrimitiveType();
  int64_t ih = 0;
  int64_t iw = 0;
  if (data_format == "NCHW") {
    spatial_dims = {2, 3};
    ih = input_shape[2];
    iw = input_shape[3];
  } else if (data_format == "NHWC") {
    spatial_dims = {1, 2};
    ih = input_shape[1];
    iw = input_shape[2];
  }
  int64_t kh = 0;
  int64_t kw = 0;
  if (adaptive) {
    std::vector<int64_t> output_spatial_size;
    for (auto dim : ksize) {
      output_spatial_size.emplace_back(static_cast<int64_t>(dim));
    }
    if (ih % output_spatial_size[0] != 0 || iw % output_spatial_size[1] != 0) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "only support: MOD(oh, ih) == 0 && MOD(ow, iw) == 0"));
      return nullptr;
    }
    kh = ih / output_spatial_size[0];
    kw = iw / output_spatial_size[1];
    strides = {kh, kw};
  } else {
    kh = static_cast<int64_t>(ksize[0]);
    kw = static_cast<int64_t>(ksize[1]);
  }
  if (padding_algorithm == "SAME") {
    auto pad_h = get_same_padding_value(ih, kh, strides[0]);
    auto pad_w = get_same_padding_value(iw, kw, strides[1]);
    padding = {pad_h[0], pad_h[1], pad_w[0], pad_w[1]};
  } else {
    if (padding.size() == 1) {
      padding = {padding[0], padding[0], padding[0], padding[0]};
    } else if (padding.size() == 2) {
      padding = {padding[0], padding[0], padding[1], padding[1]};
    } else if (padding.size() == 8) {
      if (data_format == "NCHW") {
        padding = {padding[4], padding[5], padding[6], padding[7]};
      } else if (data_format == "NHWC") {
        padding = {padding[2], padding[3], padding[4], padding[5]};
      }
    }
  }
  std::vector<int64_t> kernel_shape = {kh, kw};
  bool do_transpose =
      in.GetType().GetShape().size() == 4 && data_format == "NCHW";
  do_transpose = do_transpose && running_mode == RunningMode::ADAPTIVE;
  if (pooling_type == "max") {
    if (global_pooling) {
      PADDLE_THROW(
          platform::errors::Unimplemented("Unsupported global "
                                          "pooling_type: %s",
                                          pooling_type.c_str()));
      return nullptr;
    }
    auto grad_in = builder::MaxPool2DGrad(out_grad,
                                          in,
                                          ksize,
                                          /*ceil_mode=*/false,
                                          "NOTSET",
                                          data_format.c_str(),
                                          strides,
                                          padding);
    return std::make_shared<GcuOp>(grad_in);
  } else if (pooling_type == "avg") {
    if (global_pooling) {
      float grad_data = 1.0 / static_cast<float>(ih * iw);
      auto grad = builder::FullLike(in, grad_data);
      return std::make_shared<GcuOp>(out_grad * grad);
    }
    if (do_transpose) {
      out_grad = builder::Transpose(out_grad, {0, 2, 3, 1});
      spatial_dims = {1, 2};
      in = builder::Transpose(in, {0, 2, 3, 1});
      input_shape = in.GetType().GetShape();
      auto tmp_op =
          std::make_shared<GcuOp>(builder::AveragePoolGrad(out_grad,
                                                           spatial_dims,
                                                           input_shape,
                                                           kernel_shape,
                                                           strides,
                                                           false,
                                                           false,
                                                           padding,
                                                           in.GetType()));
      return std::make_shared<GcuOp>(builder::Transpose(*tmp_op, {0, 3, 1, 2}));
    }
    return std::make_shared<GcuOp>(builder::AveragePoolGrad(out_grad,
                                                            spatial_dims,
                                                            input_shape,
                                                            kernel_shape,
                                                            strides,
                                                            false,
                                                            false,
                                                            padding,
                                                            in.GetType()));
  } else {
    PADDLE_THROW(
        platform::errors::Unimplemented("Unsupported pooling_type:"
                                        "%s",
                                        pooling_type.c_str()));
    return nullptr;
  }
}

EQUIVALENCE_TRANS_FUNC_REG(kPool2D, INSENSITIVE, Pool2DEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kPool2DGrad,
                           INSENSITIVE,
                           Pool2DGradEquivalenceTrans);

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
