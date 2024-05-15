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

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "paddle/fluid/platform/device/gcu/register/register.h"
#include "paddle/phi/core/ddim.h"
#include "paddle/phi/core/errors.h"
#include "paddle/phi/kernels/cpu/conv_util.h"

namespace paddle {
namespace platform {
namespace gcu {

IMPLEMT_EQUIVALENCE_TRANS_CFIRST_FUNC(
    gcu_builder, node, map_inputs, map_inputs_formats, Conv3dEquivalenceTrans) {
  auto get_same_padding_value =
      [](int64_t dim, int64_t ksize, int64_t stride) -> std::vector<int64_t> {
    int64_t pad_along_dim = 0;
    if (dim % stride == 0) {
      pad_along_dim = std::max(ksize - stride, static_cast<int64_t>(0));
    } else {
      pad_along_dim = std::max(ksize - (dim % stride), static_cast<int64_t>(0));
    }
    int64_t pad_low = pad_along_dim / 2;
    int64_t pad_high = pad_along_dim - pad_low;
    std::vector<int64_t> padding{pad_low, pad_high};
    return padding;
  };

  auto* op = node->Op();
  auto groups = PADDLE_GET_CONST(int, op->GetAttr("groups"));

  PADDLE_ENFORCE_EQ(groups,
                    1,
                    platform::errors::Unimplemented(
                        "conv3d in gcu only support groups == 1 : %d", groups));

  auto padding_algorithm =
      PADDLE_GET_CONST(std::string, op->GetAttr("padding_algorithm"));
  auto data_format = PADDLE_GET_CONST(std::string, op->GetAttr("data_format"));
  auto dilations = PADDLE_GET_CONST(std::vector<int>, op->GetAttr("dilations"));
  auto strides = PADDLE_GET_CONST(std::vector<int>, op->GetAttr("strides"));
  auto paddings = PADDLE_GET_CONST(std::vector<int>, op->GetAttr("paddings"));
  std::vector<builder::Op> ops;
  int64_t group = static_cast<int64_t>(groups);

  if (map_inputs.count("Input") != 0) {
    VLOG(10) << "inputs size:" << map_inputs["Input"].size();
    auto op_ptr = map_inputs["Input"].at(0);
    auto input_shape = op_ptr->GetType().GetShape();
    if (data_format == "NCDHW") {
      ops.emplace_back(builder::Transpose(*op_ptr, {0, 2, 3, 4, 1}));
    } else {
      ops.emplace_back(*op_ptr);
    }
  } else {
    PADDLE_ENFORCE_EQ(
        true, false, platform::errors::NotFound("lack of [Input] gcu op"));
  }

  if (map_inputs.count("Filter") != 0) {
    VLOG(10) << "Filter size:" << map_inputs["Filter"].size();
    auto op_ptr = map_inputs["Filter"].at(0);
    ops.emplace_back(builder::Transpose(*op_ptr, {2, 3, 4, 1, 0}));
  } else {
    PADDLE_ENFORCE_EQ(
        true, false, platform::errors::NotFound("lack of [Filter] gcu op"));
  }

  // optional input
  if (map_inputs.count("ResidualData") != 0 &&
      map_inputs["ResidualData"].size() != 0) {
    ops.push_back(*(map_inputs["ResidualData"].at(0)));
  }

  std::vector<int64_t> stride;
  std::vector<int64_t> dilation;
  std::vector<int64_t> padding;
  for (auto dim : strides) {
    stride.emplace_back(static_cast<int64_t>(dim));
  }
  for (auto dim : dilations) {
    dilation.emplace_back(static_cast<int64_t>(dim));
  }

  PADDLE_ENFORCE_EQ(
      stride.size(),
      3,
      platform::errors::InvalidArgument("the size of stride not valid."));
  PADDLE_ENFORCE_EQ(
      dilation.size(),
      3,
      platform::errors::InvalidArgument("the size of dilation not valid."));

  for (auto dim : paddings) {
    padding.emplace_back(static_cast<int64_t>(dim));
  }

  auto input_shape = ops[0].GetType().GetShape();
  auto kernel_shape = ops[1].GetType().GetShape();
  int64_t id = input_shape[1];
  int64_t ih = input_shape[2];
  int64_t iw = input_shape[3];
  int64_t kd = kernel_shape[0];
  int64_t kh = kernel_shape[1];
  int64_t kw = kernel_shape[2];

  if (padding_algorithm == "SAME") {
    auto pad_d = get_same_padding_value(id, kd, stride[0]);
    auto pad_h = get_same_padding_value(ih, kh, stride[1]);
    auto pad_w = get_same_padding_value(iw, kw, stride[2]);
    padding = {pad_d[0], pad_d[1], pad_h[0], pad_h[1], pad_w[0], pad_w[1]};
  } else {
    if (padding.size() == 1) {
      padding = {padding[0],
                 padding[0],
                 padding[0],
                 padding[0],
                 padding[0],
                 padding[0]};
    } else if (padding.size() == 3) {
      padding = {padding[0],
                 padding[0],
                 padding[1],
                 padding[1],
                 padding[2],
                 padding[2]};
    } else if (padding.size() == 10) {
      if (data_format == "NCDHW") {
        padding = {padding[4],
                   padding[5],
                   padding[6],
                   padding[7],
                   padding[8],
                   padding[9]};
      } else if (data_format == "NDHWC") {
        padding = {padding[2],
                   padding[3],
                   padding[4],
                   padding[5],
                   padding[6],
                   padding[7]};
      }
    }
  }

  int64_t dout = (id + padding[0] + padding[1] - (dilation[0] * (kd - 1) + 1)) /
                     stride[0] +
                 1;
  int64_t hout = (ih + padding[2] + padding[3] - (dilation[1] * (kh - 1) + 1)) /
                     stride[1] +
                 1;
  int64_t wout = (iw + padding[4] + padding[5] - (dilation[2] * (kw - 1) + 1)) /
                     stride[2] +
                 1;

  std::vector<int64_t> out_shape{
      input_shape[0], dout, hout, wout, kernel_shape[4]};

  builder::ConvDimensionNumbers dnums(
      0, 0, {0, 0, 0}, 0, 0, {0, 0, 0}, 0, 0, {0, 0, 0});
  dnums.set_input_batch_dimension(0);
  dnums.set_input_spatial_dimensions({1, 2, 3});
  dnums.set_input_feature_dimension(4);
  dnums.set_output_batch_dimension(0);
  dnums.set_output_spatial_dimensions({1, 2, 3});
  dnums.set_output_feature_dimension(4);

  dnums.set_kernel_spatial_dimensions({0, 1, 2});
  dnums.set_kernel_input_feature_dimension(3);
  dnums.set_kernel_output_feature_dimension(4);

  auto resultType =
      builder::Type(out_shape, ops[0].GetType().GetPrimitiveType());

  auto out = builder::Conv(ops[0],     // builder::Op lhs
                           ops[1],     // builder::Op rhs
                           dnums,      // builder::ConvDimensionNumbers
                           stride,     // window_strides
                           padding,    // padding
                           {1, 1, 1},  // lhs_dilation
                           dilation,   // lhs_dilation
                           {},         // window_reversal
                           "",         // auto_pad
                           group,      // feature_group_count
                           1,          // batch_group_count=
                           {"DEFAULT", "DEFAULT"},  // precision_config
                           resultType);

  if (data_format == "NCDHW") {
    auto transpose = builder::Transpose(out, {0, 4, 1, 2, 3});
    return std::make_shared<GcuOp>(transpose);
  } else {
    return std::make_shared<GcuOp>(out);
  }
}

IMPLEMT_EQUIVALENCE_TRANS_CFIRST_FUNC(gcu_builder,
                                      node,
                                      map_inputs,
                                      map_inputs_formats,
                                      Conv3dGradEquivalenceTrans) {
  auto get_same_padding_value =
      [](int64_t dim, int64_t ksize, int64_t stride) -> std::vector<int64_t> {
    int64_t pad_along_dim = 0;
    if (dim % stride == 0) {
      pad_along_dim = std::max(ksize - stride, static_cast<int64_t>(0));
    } else {
      pad_along_dim = std::max(ksize - (dim % stride), static_cast<int64_t>(0));
    }
    int64_t pad_low = pad_along_dim / 2;
    int64_t pad_high = pad_along_dim - pad_low;
    std::vector<int64_t> padding{pad_low, pad_high};
    return padding;
  };

  auto get_backprop_filter_padding =
      [](int64_t input_dim,
         int64_t output_dim,
         int64_t kernel_size,
         int64_t stride,
         int64_t dilation,
         int64_t padding_before,
         const std::string& padding_algorithm) -> std::pair<int64_t, int64_t> {
    std::pair<int64_t, int64_t> padding_dim;
    int64_t expanded_output_size = (output_dim - 1) * stride + 1;
    int64_t padded_in_size = (kernel_size - 1) * dilation;
    padded_in_size += expanded_output_size;
    int64_t pad_total = padded_in_size - input_dim;
    int64_t pad_before = padding_algorithm == "EXPLICIT" ? padding_before
                         : padding_algorithm == "SAME"
                             ? std::max<int64_t>(pad_total / 2, 0)
                             : 0;
    padding_dim = {pad_before, pad_total - pad_before};
    return padding_dim;
  };

  auto get_backprop_input_padding =
      [](int64_t input_dim,
         int64_t output_dim,
         int64_t kernel_size,
         int64_t stride,
         int64_t dilation,
         int64_t padding_before) -> std::pair<int64_t, int64_t> {
    std::pair<int64_t, int64_t> padding_dim;
    int64_t effective_filter_size = (kernel_size - 1) * dilation + 1;
    int64_t expanded_output_size = (output_dim - 1) * stride + 1;
    int64_t padded_out_size = input_dim + effective_filter_size - 1;
    int64_t pad_before = effective_filter_size - 1 - padding_before;
    int64_t pad_after = padded_out_size - expanded_output_size - pad_before;
    padding_dim = {pad_before, pad_after};
    return padding_dim;
  };

  auto* op = node->Op();

  auto data_format = PADDLE_GET_CONST(std::string, op->GetAttr("data_format"));

  builder::Op out_grad = *(map_inputs["Output@GRAD"].at(0));
  builder::Op input = *(map_inputs["Input"].at(0));
  builder::Op filter = *(map_inputs["Filter"].at(0));
  if (data_format == "NCDHW") {
    out_grad = builder::Transpose(out_grad, {0, 2, 3, 4, 1});
    input = builder::Transpose(input, {0, 2, 3, 4, 1});
  }
  filter = builder::Transpose(filter, {2, 3, 4, 1, 0});

  auto groups = PADDLE_GET_CONST(int, op->GetAttr("groups"));

  PADDLE_ENFORCE_EQ(groups,
                    1,
                    platform::errors::Unimplemented(
                        "conv3d in gcu only support groups == 1 : %d", groups));

  auto padding_algorithm =
      PADDLE_GET_CONST(std::string, op->GetAttr("padding_algorithm"));
  auto dilations = PADDLE_GET_CONST(std::vector<int>, op->GetAttr("dilations"));
  auto strides = PADDLE_GET_CONST(std::vector<int>, op->GetAttr("strides"));
  auto paddings = PADDLE_GET_CONST(std::vector<int>, op->GetAttr("paddings"));
  int64_t group = static_cast<int64_t>(groups);

  std::vector<int64_t> stride;
  std::vector<int64_t> dilation;
  std::vector<int64_t> padding;
  for (auto dim : strides) {
    stride.emplace_back(static_cast<int64_t>(dim));
  }
  for (auto dim : dilations) {
    dilation.emplace_back(static_cast<int64_t>(dim));
  }

  PADDLE_ENFORCE_EQ(
      stride.size(),
      3,
      platform::errors::InvalidArgument("the size of stride not valid."));
  PADDLE_ENFORCE_EQ(
      dilation.size(),
      3,
      platform::errors::InvalidArgument("the size of dilation not valid."));

  for (auto dim : paddings) {
    padding.emplace_back(static_cast<int64_t>(dim));
  }

  auto input_shape = input.GetType().GetShape();
  auto kernel_shape = filter.GetType().GetShape();
  auto output_shape = out_grad.GetType().GetShape();
  int64_t id = input_shape[1];
  int64_t ih = input_shape[2];
  int64_t iw = input_shape[3];
  int64_t kd = kernel_shape[0];
  int64_t kh = kernel_shape[1];
  int64_t kw = kernel_shape[2];
  int64_t od = output_shape[1];
  int64_t oh = output_shape[2];
  int64_t ow = output_shape[3];
  if (padding_algorithm == "SAME") {
    auto pad_d = get_same_padding_value(id, kd, stride[0]);
    auto pad_h = get_same_padding_value(ih, kh, stride[1]);
    auto pad_w = get_same_padding_value(iw, kw, stride[2]);
    padding = {pad_d[0], pad_d[1], pad_h[0], pad_h[1], pad_w[0], pad_w[1]};
  } else {
    if (padding.size() == 1) {
      padding = {padding[0],
                 padding[0],
                 padding[0],
                 padding[0],
                 padding[0],
                 padding[0]};
    } else if (padding.size() == 3) {
      padding = {padding[0],
                 padding[0],
                 padding[1],
                 padding[1],
                 padding[2],
                 padding[2]};
    } else if (padding.size() == 10) {
      if (data_format == "NCDHW") {
        padding = {padding[4],
                   padding[5],
                   padding[6],
                   padding[7],
                   padding[8],
                   padding[9]};
      } else if (data_format == "NDHWC") {
        padding = {padding[2],
                   padding[3],
                   padding[4],
                   padding[5],
                   padding[6],
                   padding[7]};
      }
    }
  }
  auto output_name_map = op->Outputs();
  builder::Op filter_grad;
  if (output_name_map.count("Filter@GRAD") != 0 &&
      output_name_map["Filter@GRAD"].size() > 0) {
    auto pad_d = get_backprop_filter_padding(
        id, od, kd, stride[0], dilation[0], padding[0], padding_algorithm);
    auto pad_h = get_backprop_filter_padding(
        ih, oh, kh, stride[1], dilation[1], padding[2], padding_algorithm);
    auto pad_w = get_backprop_filter_padding(
        iw, ow, kw, stride[2], dilation[2], padding[4], padding_algorithm);

    std::vector<std::vector<int64_t>> paddings = {{pad_d.first, pad_d.second},
                                                  {pad_h.first, pad_h.second},
                                                  {pad_w.first, pad_w.second}};

    builder::ConvDimensionNumbers dnums(
        0, 0, {0, 0, 0}, 0, 0, {0, 0, 0}, 0, 0, {0, 0, 0});

    dnums.set_input_spatial_dimensions({1, 2, 3});
    dnums.set_output_feature_dimension(4);
    dnums.set_input_batch_dimension(4);
    dnums.set_input_feature_dimension(0);
    dnums.set_output_batch_dimension(3);
    dnums.set_output_spatial_dimensions({0, 1, 2});

    dnums.set_kernel_output_feature_dimension(4);
    dnums.set_kernel_spatial_dimensions({1, 2, 3});
    dnums.set_kernel_input_feature_dimension(0);

    filter_grad = builder::Conv(input,
                                out_grad,
                                dnums,
                                /*window_strides=*/dilation,
                                /*padding=*/paddings,
                                /*lhs_dilation=*/{1, 1, 1},
                                /*rhs_dilation=*/stride,
                                /*window_reversal=*/{},
                                /*auto_pad=*/"",
                                /*feature_group_count=*/1,
                                /*batch_group_count=*/group,
                                /*precision_config=*/{"DEFAULT", "DEFAULT"},
                                filter.GetType());
    filter_grad.SetAttribute("op_type",
                             builder::Attribute("Conv3DBackpropFilter"));

    filter_grad = builder::Transpose(filter_grad, {4, 3, 0, 1, 2});
  }
  if (output_name_map.count("Input@GRAD") != 0 &&
      output_name_map["Input@GRAD"].size() == 0) {
    return std::make_shared<GcuOp>(filter_grad);
  }

  builder::Op input_grad;
  if (output_name_map.count("Input@GRAD") != 0 &&
      output_name_map["Input@GRAD"].size() > 0) {
    auto filter_reverse = builder::Reverse(filter, {0, 1, 2}, filter.GetType());
    std::vector<int64_t> lhs_dilation = stride;
    std::vector<int64_t> rhs_dilation = dilation;

    builder::ConvDimensionNumbers dnums(
        0, 0, {0, 0, 0}, 0, 0, {0, 0, 0}, 0, 0, {0, 0, 0});

    dnums.set_input_batch_dimension(0);
    dnums.set_input_spatial_dimensions({1, 2, 3});
    dnums.set_input_feature_dimension(4);
    dnums.set_output_batch_dimension(0);
    dnums.set_output_spatial_dimensions({1, 2, 3});
    dnums.set_output_feature_dimension(4);

    dnums.set_kernel_spatial_dimensions({0, 1, 2});
    dnums.set_kernel_input_feature_dimension(4);
    dnums.set_kernel_output_feature_dimension(3);

    auto pad_d = get_backprop_input_padding(
        id, od, kd, stride[0], dilation[0], padding[0]);
    auto pad_h = get_backprop_input_padding(
        ih, oh, kh, stride[1], dilation[1], padding[2]);
    auto pad_w = get_backprop_input_padding(
        iw, ow, kw, stride[2], dilation[2], padding[4]);
    std::vector<std::vector<int64_t>> paddings = {{pad_d.first, pad_d.second},
                                                  {pad_h.first, pad_h.second},
                                                  {pad_w.first, pad_w.second}};

    input_grad = builder::Conv(out_grad,
                               filter_reverse,
                               dnums,
                               /*window_strides=*/{1, 1, 1},
                               /*padding=*/paddings,
                               /*lhs_dilation=*/lhs_dilation,
                               /*rhs_dilation=*/rhs_dilation,
                               /*window_reversal=*/{},
                               /*auto_pad=*/"",
                               /*feature_group_count=*/1,
                               /*batch_group_count=*/1,
                               /*precision_config=*/{},
                               input.GetType());
    input_grad.SetAttribute("op_type",
                            builder::Attribute("Conv3DBackpropInput"));

    if (data_format == "NCDHW") {
      input_grad = builder::Transpose(input_grad, {0, 4, 1, 2, 3});
    }
  }

  std::vector<builder::Op> outputs{filter_grad, input_grad};
  std::vector<std::string> output_names{"Filter@GRAD", "Input@GRAD"};
  std::string output_names_attr(output_name_map[output_names[0]][0]);
  for (size_t i = 1; i < output_names.size(); ++i) {
    output_names_attr += ";" + output_name_map[output_names[i]][0];
  }
  std::vector<builder::PrimitiveType> tuple_dtype;
  std::vector<std::vector<int64_t>> tuple_shape;
  for (uint i = 0; i < outputs.size(); i++) {
    tuple_shape.push_back(outputs[i].GetType().GetShape());
    tuple_dtype.push_back(outputs[i].GetType().GetPrimitiveType());
  }
  builder::Type outputs_type(tuple_shape, tuple_dtype);
  auto result = builder::Tuple(outputs, outputs_type);
  result.SetAttribute(kAttrOpOutVarName,
                      builder::Attribute(output_names_attr.c_str()));
  return std::make_shared<GcuOp>(result);
}

IMPLEMT_EQUIVALENCE_TRANS_CFIRST_FUNC(gcu_builder,
                                      node,
                                      map_inputs,
                                      map_inputs_formats,
                                      Conv3dTransposeEquivalenceTrans) {
  auto get_same_padding_value =
      [](int64_t dim, int64_t ksize, int64_t stride) -> std::vector<int64_t> {
    int64_t pad_along_dim = 0;
    if (dim % stride == 0) {
      pad_along_dim = std::max(ksize - stride, static_cast<int64_t>(0));
    } else {
      pad_along_dim = std::max(ksize - (dim % stride), static_cast<int64_t>(0));
    }
    int64_t pad_low = pad_along_dim / 2;
    int64_t pad_high = pad_along_dim - pad_low;
    std::vector<int64_t> padding{pad_low, pad_high};
    return padding;
  };

  auto get_conv3d_transpose_padding =
      [](const std::vector<int64_t>& input_spatial_dims,
         const std::vector<int64_t>& output_spatial_dims,
         const std::vector<int64_t>& ksize,
         const std::vector<int64_t>& stride,
         const std::vector<int64_t>& dilation,
         const std::vector<int64_t>& padding,
         const std::vector<int64_t>& output_padding,
         const std::string& auto_pad) -> std::vector<int64_t> {
    std::vector<int64_t> padding_value;
    for (size_t i = 0; i < input_spatial_dims.size(); ++i) {
      int64_t expanded_input_size = (input_spatial_dims[i] - 1) * stride[i] + 1;
      int64_t effective_filter_size = (ksize[i] - 1) * dilation[i] + 1;
      int64_t pad_before = effective_filter_size - 1 - padding[i * 2];
      int64_t padded_out_size =
          output_spatial_dims[i] + effective_filter_size - 1;
      int64_t pad_after = padded_out_size - expanded_input_size - pad_before;
      padding_value.emplace_back(pad_before);
      padding_value.emplace_back(pad_after);
    }
    return padding_value;
  };

  auto GetConvTransposeDim = [](int64_t input_dim,
                                int64_t ksize,
                                int64_t stride,
                                int64_t dilation,
                                int64_t pad_low,
                                int64_t pad_high,
                                int64_t output_padding) {
    int64_t expanded_input_size = (input_dim - 1) * stride + 1;
    int64_t effective_filter_size = (ksize - 1) * dilation + 1;
    int64_t output_dim = expanded_input_size - 1 + output_padding +
                         effective_filter_size - pad_low - pad_high;
    return output_dim;
  };

  auto* op = node->Op();
  auto groups = PADDLE_GET_CONST(int, op->GetAttr("groups"));

  PADDLE_ENFORCE_EQ(groups,
                    1,
                    platform::errors::Unimplemented(
                        "conv3d in gcu only support groups == 1 : %d", groups));

  auto padding_algorithm =
      PADDLE_GET_CONST(std::string, op->GetAttr("padding_algorithm"));
  auto data_format = PADDLE_GET_CONST(std::string, op->GetAttr("data_format"));
  auto dilations = PADDLE_GET_CONST(std::vector<int>, op->GetAttr("dilations"));
  auto strides = PADDLE_GET_CONST(std::vector<int>, op->GetAttr("strides"));
  auto paddings = PADDLE_GET_CONST(std::vector<int>, op->GetAttr("paddings"));
  auto output_paddings =
      PADDLE_GET_CONST(std::vector<int>, op->GetAttr("output_padding"));
  std::vector<builder::Op> ops;
  int64_t group = static_cast<int64_t>(groups);

  if (map_inputs.count("Input") != 0) {
    VLOG(10) << "inputs size:" << map_inputs["Input"].size();
    auto op_ptr = map_inputs["Input"].at(0);
    auto input_shape = op_ptr->GetType().GetShape();
    if (data_format == "NCHW") {
      ops.emplace_back(builder::Transpose(*op_ptr, {0, 2, 3, 4, 1}));
    } else {
      ops.emplace_back(*op_ptr);
    }
  } else {
    PADDLE_ENFORCE_EQ(
        true, false, platform::errors::NotFound("lack of [Input] gcu op"));
  }

  if (map_inputs.count("Filter") != 0) {
    VLOG(10) << "Filter size:" << map_inputs["Filter"].size();
    auto op_ptr = map_inputs["Filter"].at(0);
    auto k = builder::Transpose(*op_ptr, {2, 3, 4, 1, 0});
    ops.emplace_back(builder::Reverse(k, {0, 1, 2}, k.GetType()));
  } else {
    PADDLE_ENFORCE_EQ(
        true, false, platform::errors::NotFound("lack of [Filter] gcu op"));
  }

  // optional input
  if (map_inputs.count("ResidualData") != 0 &&
      map_inputs["ResidualData"].size() != 0) {
    ops.push_back(*(map_inputs["ResidualData"].at(0)));
  }

  std::vector<int64_t> stride;
  std::vector<int64_t> dilation;
  std::vector<int64_t> padding;
  std::vector<int64_t> output_padding;
  std::vector<int64_t> output_shape;
  for (auto dim : strides) {
    stride.emplace_back(static_cast<int64_t>(dim));
  }
  for (auto dim : dilations) {
    dilation.emplace_back(static_cast<int64_t>(dim));
  }
  for (auto dim : output_paddings) {
    output_padding.emplace_back(static_cast<int64_t>(dim));
  }

  PADDLE_ENFORCE_EQ(
      stride.size(),
      3,
      platform::errors::InvalidArgument("the size of stride not valid."));
  PADDLE_ENFORCE_EQ(
      dilation.size(),
      3,
      platform::errors::InvalidArgument("the size of dilation not valid."));

  for (auto dim : paddings) {
    padding.emplace_back(static_cast<int64_t>(dim));
  }
  auto input_shape = ops[0].GetType().GetShape();
  auto kernel_shape = ops[1].GetType().GetShape();
  int64_t id = input_shape[1];
  int64_t ih = input_shape[2];
  int64_t iw = input_shape[3];
  int64_t kd = kernel_shape[0];
  int64_t kh = kernel_shape[1];
  int64_t kw = kernel_shape[2];

  if (padding_algorithm == "SAME") {
    auto pad_d = get_same_padding_value(id, kd, stride[0]);
    auto pad_h = get_same_padding_value(ih, kh, stride[1]);
    auto pad_w = get_same_padding_value(iw, kw, stride[2]);
    padding = {pad_d[0], pad_d[1], pad_h[0], pad_h[1], pad_w[0], pad_w[1]};
  } else {
    if (padding.size() == 1) {
      padding = {padding[0],
                 padding[0],
                 padding[0],
                 padding[0],
                 padding[0],
                 padding[0]};
    } else if (padding.size() == 3) {
      padding = {padding[0],
                 padding[0],
                 padding[1],
                 padding[1],
                 padding[2],
                 padding[2]};
    } else if (padding.size() == 10) {
      if (data_format == "NCHW") {
        padding = {padding[4],
                   padding[5],
                   padding[6],
                   padding[7],
                   padding[8],
                   padding[9]};
      } else if (data_format == "NHWC") {
        padding = {padding[2],
                   padding[3],
                   padding[4],
                   padding[5],
                   padding[6],
                   padding[7]};
      }
    }
  }

  if (output_padding.size() == 0) {
    output_padding = {0, 0, 0};
  } else if (output_padding.size() == 1) {
    output_padding = {output_padding[0], output_padding[0], output_padding[0]};
  }

  auto od = GetConvTransposeDim(id,
                                kd,
                                stride[0],
                                dilation[0],
                                padding[0],
                                padding[1],
                                output_padding[0]);
  auto oh = GetConvTransposeDim(ih,
                                kh,
                                stride[1],
                                dilation[1],
                                padding[2],
                                padding[3],
                                output_padding[1]);
  auto ow = GetConvTransposeDim(iw,
                                kw,
                                stride[2],
                                dilation[2],
                                padding[4],
                                padding[5],
                                output_padding[2]);

  auto real_padding = get_conv3d_transpose_padding({id, ih, iw},
                                                   {od, oh, ow},
                                                   {kd, kh, kw},
                                                   stride,
                                                   dilation,
                                                   padding,
                                                   output_padding,
                                                   "NOTSET");

  builder::ConvDimensionNumbers dims_attr(
      0, 4, {1, 2, 3}, 4, 3, {0, 1, 2}, 0, 4, {1, 2, 3});

  std::vector<int64_t> out_shape{input_shape[0], od, oh, ow, kernel_shape[3]};
  auto resultType =
      builder::Type(out_shape, ops[0].GetType().GetPrimitiveType());

  auto conv2d_transpose = builder::Conv(ops[0],
                                        ops[1],
                                        dims_attr,
                                        {1, 1, 1},
                                        real_padding,
                                        stride,
                                        dilation,
                                        {},
                                        "",
                                        group,
                                        1,
                                        {});

  conv2d_transpose.SetAttribute("op_type",
                                builder::Attribute("Conv3DBackpropInput"));
  if (data_format == "NCHW") {
    auto transpose = builder::Transpose(conv2d_transpose, {0, 4, 1, 2, 3});
    return std::make_shared<GcuOp>(transpose);
  } else {
    return std::make_shared<GcuOp>(conv2d_transpose);
  }
}

IMPLEMT_EQUIVALENCE_TRANS_CFIRST_FUNC(gcu_builder,
                                      node,
                                      map_inputs,
                                      map_inputs_formats,
                                      Conv3dTransposeGradEquivalenceTrans) {
  auto get_same_padding_value =
      [](int64_t dim, int64_t ksize, int64_t stride) -> std::vector<int64_t> {
    int64_t pad_along_dim = 0;
    if (dim % stride == 0) {
      pad_along_dim = std::max(ksize - stride, static_cast<int64_t>(0));
    } else {
      pad_along_dim = std::max(ksize - (dim % stride), static_cast<int64_t>(0));
    }
    int64_t pad_low = pad_along_dim / 2;
    int64_t pad_high = pad_along_dim - pad_low;
    std::vector<int64_t> padding{pad_low, pad_high};
    return padding;
  };

  auto get_backprop_filter_padding =
      [](int64_t input_dim,
         int64_t output_dim,
         int64_t kernel_size,
         int64_t stride,
         int64_t dilation,
         int64_t padding_before,
         const std::string& padding_algorithm) -> std::pair<int64_t, int64_t> {
    std::pair<int64_t, int64_t> padding_dim;
    int64_t expanded_output_size = (output_dim - 1) * stride + 1;
    int64_t padded_in_size = (kernel_size - 1) * dilation;
    padded_in_size += expanded_output_size;
    int64_t pad_total = padded_in_size - input_dim;
    int64_t pad_before = padding_algorithm == "EXPLICIT" ? padding_before
                         : padding_algorithm == "SAME"
                             ? std::max<int64_t>(pad_total / 2, 0)
                             : 0;
    padding_dim = {pad_before, pad_total - pad_before};
    return padding_dim;
  };

  auto* op = node->Op();
  auto groups = PADDLE_GET_CONST(int, op->GetAttr("groups"));

  PADDLE_ENFORCE_EQ(groups,
                    1,
                    platform::errors::Unimplemented(
                        "conv3d in gcu only support groups == 1 : %d", groups));

  auto padding_algorithm =
      PADDLE_GET_CONST(std::string, op->GetAttr("padding_algorithm"));
  auto data_format = PADDLE_GET_CONST(std::string, op->GetAttr("data_format"));
  auto dilations = PADDLE_GET_CONST(std::vector<int>, op->GetAttr("dilations"));
  auto strides = PADDLE_GET_CONST(std::vector<int>, op->GetAttr("strides"));
  auto paddings = PADDLE_GET_CONST(std::vector<int>, op->GetAttr("paddings"));
  auto output_paddings =
      PADDLE_GET_CONST(std::vector<int>, op->GetAttr("output_padding"));

  std::vector<int64_t> stride;
  std::vector<int64_t> dilation;
  std::vector<int64_t> padding;
  std::vector<int64_t> output_padding;
  for (auto dim : strides) {
    stride.emplace_back(static_cast<int64_t>(dim));
  }
  for (auto dim : dilations) {
    dilation.emplace_back(static_cast<int64_t>(dim));
  }
  for (auto dim : output_paddings) {
    output_padding.emplace_back(static_cast<int64_t>(dim));
  }

  PADDLE_ENFORCE_EQ(
      stride.size(),
      3,
      platform::errors::InvalidArgument("the size of stride not valid."));
  PADDLE_ENFORCE_EQ(
      dilation.size(),
      3,
      platform::errors::InvalidArgument("the size of dilation not valid."));

  for (auto dim : paddings) {
    padding.emplace_back(static_cast<int64_t>(dim));
  }

  builder::Op out_grad = *(map_inputs["Output@GRAD"].at(0));
  builder::Op input = *(map_inputs["Input"].at(0));
  builder::Op filter = *(map_inputs["Filter"].at(0));
  if (data_format == "NCHW") {
    out_grad = builder::Transpose(out_grad, {0, 2, 3, 4, 1});
    input = builder::Transpose(input, {0, 2, 3, 4, 1});
  }
  filter = builder::Transpose(filter, {2, 3, 4, 1, 0});

  auto input_shape = input.GetType().GetShape();
  auto kernel_shape = filter.GetType().GetShape();
  auto output_shape = out_grad.GetType().GetShape();
  int64_t id = input_shape[1];
  int64_t ih = input_shape[2];
  int64_t iw = input_shape[3];
  int64_t kd = kernel_shape[0];
  int64_t kh = kernel_shape[1];
  int64_t kw = kernel_shape[2];
  int64_t od = output_shape[1];
  int64_t oh = output_shape[2];
  int64_t ow = output_shape[3];

  if (padding_algorithm == "SAME") {
    auto pad_d = get_same_padding_value(id, kd, stride[0]);
    auto pad_h = get_same_padding_value(ih, kh, stride[1]);
    auto pad_w = get_same_padding_value(iw, kw, stride[2]);
    padding = {pad_d[0], pad_d[1], pad_h[0], pad_h[1], pad_w[0], pad_w[1]};
  } else {
    if (padding.size() == 1) {
      padding = {padding[0],
                 padding[0],
                 padding[0],
                 padding[0],
                 padding[0],
                 padding[0]};
    } else if (padding.size() == 3) {
      padding = {padding[0],
                 padding[0],
                 padding[1],
                 padding[1],
                 padding[2],
                 padding[2]};
    } else if (padding.size() == 10) {
      if (data_format == "NCHW") {
        padding = {padding[4],
                   padding[5],
                   padding[6],
                   padding[7],
                   padding[8],
                   padding[9]};
      } else if (data_format == "NHWC") {
        padding = {padding[2],
                   padding[3],
                   padding[4],
                   padding[5],
                   padding[6],
                   padding[7]};
      }
    }
  }

  auto output_name_map = op->Outputs();

  builder::Op filter_grad;
  if (output_name_map.count("Filter@GRAD") != 0 &&
      output_name_map["Filter@GRAD"].size() > 0) {
    auto pad_d = get_backprop_filter_padding(
        od, id, kd, stride[0], dilation[0], padding[0], padding_algorithm);
    auto pad_h = get_backprop_filter_padding(
        oh, ih, kh, stride[1], dilation[1], padding[2], padding_algorithm);
    auto pad_w = get_backprop_filter_padding(
        ow, iw, kw, stride[2], dilation[2], padding[4], padding_algorithm);

    std::vector<std::vector<int64_t>> paddings = {{pad_d.first, pad_d.second},
                                                  {pad_h.first, pad_h.second},
                                                  {pad_w.first, pad_w.second}};

    builder::ConvDimensionNumbers dims_attr(
        4, 0, {1, 2, 3}, 0, 4, {1, 2, 3}, 3, 4, {0, 1, 2});
    filter_grad = builder::Conv(out_grad,
                                input,
                                dims_attr,
                                /*window_strides=*/dilation,
                                /*padding=*/paddings,
                                /*lhs_dilation=*/{1, 1, 1},
                                /*rhs_dilation=*/stride,
                                /*window_reversal=*/{},
                                /*auto_pad=*/"",
                                /*feature_group_count=*/1,
                                /*batch_group_count=*/1,
                                /*precision_config=*/{});
    filter_grad.SetAttribute(
        "op_type", builder::Attribute("Conv3DTransposeBackpropFilter"));
    filter_grad = builder::Transpose(filter_grad, {4, 3, 0, 1, 2});
  }

  if (output_name_map.count("Input@GRAD") != 0 &&
      output_name_map["Input@GRAD"].size() == 0) {
    return std::make_shared<GcuOp>(filter_grad);
  }
  builder::Op input_grad;
  if (output_name_map.count("Input@GRAD") != 0 &&
      output_name_map["Input@GRAD"].size() > 0) {
    std::vector<int64_t> lhs_dilation = {1, 1, 1};
    std::vector<int64_t> rhs_dilation = dilation;
    builder::ConvDimensionNumbers dims_attr(
        0, 4, {1, 2, 3}, 3, 4, {0, 1, 2}, 0, 4, {1, 2, 3});

    input_grad = builder::Conv(out_grad,
                               filter,
                               dims_attr,
                               /*window_strides=*/stride,
                               /*padding=*/padding,
                               /*lhs_dilation=*/lhs_dilation,
                               /*rhs_dilation=*/rhs_dilation,
                               /*window_reversal=*/{},
                               /*auto_pad=*/"",
                               /*feature_group_count=*/1,
                               /*batch_group_count=*/1,
                               /*precision_config=*/{});
    input_grad.SetAttribute("op_type",
                            builder::Attribute("Conv3DTransposeBackpropInput"));
    if (data_format == "NCHW") {
      input_grad = builder::Transpose(input_grad, {0, 4, 1, 2, 3});
    }
  }

  std::vector<builder::Op> outputs{filter_grad, input_grad};
  std::vector<std::string> output_names{"Filter@GRAD", "Input@GRAD"};
  std::string output_names_attr(output_name_map[output_names[0]][0]);
  for (size_t i = 1; i < output_names.size(); ++i) {
    output_names_attr += ";" + output_name_map[output_names[i]][0];
  }
  std::vector<builder::PrimitiveType> tuple_dtype;
  std::vector<std::vector<int64_t>> tuple_shape;
  for (uint i = 0; i < outputs.size(); i++) {
    tuple_shape.push_back(outputs[i].GetType().GetShape());
    tuple_dtype.push_back(outputs[i].GetType().GetPrimitiveType());
  }
  builder::Type outputs_type(tuple_shape, tuple_dtype);
  auto result = builder::Tuple(outputs, outputs_type);
  result.SetAttribute(kAttrOpOutVarName,
                      builder::Attribute(output_names_attr.c_str()));
  return std::make_shared<GcuOp>(result);
}

EQUIVALENCE_TRANS_FUNC_REG(kConv3D, CHANNELFIRST, Conv3dEquivalenceTrans);

EQUIVALENCE_TRANS_FUNC_REG(kConv3DGrad,
                           CHANNELFIRST,
                           Conv3dGradEquivalenceTrans);

EQUIVALENCE_TRANS_FUNC_REG(kConv3DTranspose,
                           CHANNELFIRST,
                           Conv3dTransposeEquivalenceTrans);

EQUIVALENCE_TRANS_FUNC_REG(kConv3DTransposeGrad,
                           CHANNELFIRST,
                           Conv3dTransposeGradEquivalenceTrans);

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
