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
#include "paddle/fluid/platform/device/gcu/utils/layout.h"

namespace paddle {
namespace platform {
namespace gcu {

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, Conv2DEquivalenceTrans) {
  auto* op = node->Op();
  auto groups = PADDLE_GET_CONST(int, op->GetAttr("groups"));
  auto padding_algorithm =
      PADDLE_GET_CONST(std::string, op->GetAttr("padding_algorithm"));
  auto data_format = PADDLE_GET_CONST(std::string, op->GetAttr("data_format"));
  auto dilations = PADDLE_GET_CONST(std::vector<int>, op->GetAttr("dilations"));
  auto strides = PADDLE_GET_CONST(std::vector<int>, op->GetAttr("strides"));
  auto paddings = PADDLE_GET_CONST(std::vector<int>, op->GetAttr("paddings"));
  std::vector<builder::Op> ops;
  int64_t group = static_cast<int64_t>(groups);
  // sorted_archetype_names :"Input", "Filter", "Bias", "ResidualData"
  // necessary input
  if (map_inputs.count("Input") != 0) {
    VLOG(10) << "inputs size:" << map_inputs["Input"].size();
    auto op_ptr = map_inputs["Input"].at(0);
    auto input_shape = op_ptr->GetType().GetShape();
    if (data_format == "NCHW") {
      ops.emplace_back(builder::Transpose(*op_ptr, {0, 2, 3, 1}));
    } else {
      ops.emplace_back(*op_ptr);
    }
  } else {
    PADDLE_ENFORCE_EQ(
        true, false, platform::errors::NotFound("lack of [Input] gcu op"));
  }
  // is_depth_wise = ((group > 1) && (group == channel_num));
  if (map_inputs.count("Filter") != 0) {
    VLOG(10) << "Filter size:" << map_inputs["Filter"].size();
    auto op_ptr = map_inputs["Filter"].at(0);
    if (running_mode == RunningMode::ADAPTIVE) {
      ops.emplace_back(*op_ptr);
    } else {
      ops.emplace_back(builder::Transpose(*op_ptr, {2, 3, 1, 0}));
    }
  } else {
    PADDLE_ENFORCE_EQ(
        true, false, platform::errors::NotFound("lack of [Filter] gcu op"));
  }
  // optional input
  if (map_inputs.count("Bias") != 0 && map_inputs["Bias"].size() != 0) {
    ops.push_back(*(map_inputs["Bias"].at(0)));
  }
  if (map_inputs.count("ResidualData") != 0 &&
      map_inputs["ResidualData"].size() != 0) {
    ops.push_back(*(map_inputs["ResidualData"].at(0)));
  }
  VLOG(10) << "input op number:" << ops.size();
  std::vector<int64_t> stride;
  std::vector<int64_t> dilation;
  std::vector<int64_t> padding;
  for (auto dim : strides) {
    stride.emplace_back(static_cast<int64_t>(dim));
  }
  for (auto dim : dilations) {
    dilation.emplace_back(static_cast<int64_t>(dim));
  }
  for (auto dim : paddings) {
    padding.emplace_back(static_cast<int64_t>(dim));
  }
  if (padding_algorithm == "SAME") {
    auto input_shape = ops[0].GetType().GetShape();
    auto kernel_shape = ops[1].GetType().GetShape();
    int64_t ih = input_shape[1];
    int64_t iw = input_shape[2];
    int64_t kh = kernel_shape[0];
    int64_t kw = kernel_shape[1];
    auto pad_h = get_same_padding_value(ih, kh, stride[0]);
    auto pad_w = get_same_padding_value(iw, kw, stride[1]);
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
  auto conv2d = builder::Conv2D(ops,
                                group,
                                "NOTSET",  // auto_pad
                                "NHWC",    // layout
                                stride,
                                padding,
                                dilation);

  conv2d.SetAttribute("op_type", builder::Attribute("Conv2DInference"));
  if (group > 1) {
    auto output_name_map = op->Outputs();
    auto input_name_map = op->Inputs();
    for (auto item : input_name_map) {
      std::cout << "---- depthwise conv: " << item.first << ": ";
      if (item.second.size() > 0) {
        std::cout << item.second[0];
      }
      std::cout << std::endl;
    }
  }
  if (data_format == "NCHW") {
    auto transpose = builder::Transpose(conv2d, {0, 3, 1, 2});
    return std::make_shared<GcuOp>(transpose);
  } else {
    return std::make_shared<GcuOp>(conv2d);
  }
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, Conv2DGradEquivalenceTrans) {
  // do not take bias in account because bias_grad will be calculated in
  // elementwise_add_grad input keys: Output@GRAD, Filter, Input output keys:
  // Filter@GRAD, Input@GRAD
  auto* op = node->Op();
  auto data_format = PADDLE_GET_CONST(std::string, op->GetAttr("data_format"));
  // here we only support layout of NCHW
  if (data_format != "NCHW") {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Only support NCHW for now, but got : %s", data_format.c_str()));
  }
  auto groups = PADDLE_GET_CONST(int, op->GetAttr("groups"));
  auto padding_algorithm =
      PADDLE_GET_CONST(std::string, op->GetAttr("padding_algorithm"));
  auto dilations_i32 =
      PADDLE_GET_CONST(std::vector<int>, op->GetAttr("dilations"));
  auto strides_i32 = PADDLE_GET_CONST(std::vector<int>, op->GetAttr("strides"));
  auto paddings_i32 =
      PADDLE_GET_CONST(std::vector<int>, op->GetAttr("paddings"));
  int64_t group = static_cast<int64_t>(groups);
  int64_t channel_num = map_inputs["Input"].at(0)->GetType().GetShape().at(1);
  bool depthwise = false;
  if (group > 1 && group == channel_num) {
    depthwise = true;
  }
  std::vector<int64_t> stride;
  std::vector<int64_t> dilation;
  std::vector<int64_t> padding;
  for (auto dim : strides_i32) {
    stride.emplace_back(static_cast<int64_t>(dim));
  }
  for (auto dim : dilations_i32) {
    dilation.emplace_back(static_cast<int64_t>(dim));
  }
  for (auto dim : paddings_i32) {
    padding.emplace_back(static_cast<int64_t>(dim));
  }
  builder::Op out_grad =
      builder::Transpose(*(map_inputs["Output@GRAD"].at(0)), {0, 2, 3, 1});
  builder::Op input =
      builder::Transpose(*(map_inputs["Input"].at(0)), {0, 2, 3, 1});
  builder::Op filter;
  if (running_mode == RunningMode::ADAPTIVE) {
    filter = *(map_inputs["Filter"].at(0));
    // when run, it will do transpose for kernel in host, so here is not insert
    // transpose for transpose erase, record weight name
    // TransformUtil::TransAndRecordWeights(node, "Filter",
    // filter.GetType().GetShape(), GcuLayout::NCHW, GcuLayout::HWCN);
    if (depthwise) {
      auto filter_shape = filter.GetType().GetShape();
      std::vector<int64_t> new_filter_shape = {
          filter_shape[0], filter_shape[1], filter_shape[3], filter_shape[2]};
      builder::Type new_type(new_filter_shape,
                             filter.GetType().GetPrimitiveType());
      filter = builder::Reshape(filter, new_type);
    }
  } else {
    if (depthwise)
      filter = builder::Transpose(*(map_inputs["Filter"].at(0)), {2, 3, 0, 1});
    else
      filter = builder::Transpose(*(map_inputs["Filter"].at(0)), {2, 3, 1, 0});
  }
  auto input_shape = input.GetType().GetShape();
  auto kernel_shape = filter.GetType().GetShape();
  auto output_shape = out_grad.GetType().GetShape();
  int64_t ih = input_shape[1];
  int64_t iw = input_shape[2];
  int64_t kh = kernel_shape[0];
  int64_t kw = kernel_shape[1];
  int64_t oh = output_shape[1];
  int64_t ow = output_shape[2];

  if (groups > 1 && !depthwise) {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Only support groups == 1 or depthwise conv2d, but got : %d", groups));
  }
  if (padding_algorithm == "SAME") {
    auto pad_h = get_same_padding_value(ih, kh, stride[0]);
    auto pad_w = get_same_padding_value(iw, kw, stride[1]);
    padding = {pad_h[0], pad_h[1], pad_w[0], pad_w[1]};
  } else {
    if (padding.size() == 1) {
      padding = {padding[0], padding[0], padding[0], padding[0]};
    } else if (padding.size() == 2) {
      padding = {padding[0], padding[0], padding[1], padding[1]};
    } else if (padding.size() == 8) {
      padding = {padding[4], padding[5], padding[6], padding[7]};
    }
  }
  auto output_name_map = op->Outputs();
  // calculate filter_grad
  builder::Op filter_grad;
  if (output_name_map.count("Filter@GRAD") != 0 &&
      output_name_map["Filter@GRAD"].size() > 0) {
    std::vector<int64_t> window_strides = dilation;
    std::vector<int64_t> rhs_dilation = stride;
    auto pad_h = get_backprop_filter_padding(
        ih, oh, kh, stride[0], dilation[0], padding[0], padding_algorithm);
    auto pad_w = get_backprop_filter_padding(
        iw, ow, kw, stride[1], dilation[1], padding[2], padding_algorithm);
    std::vector<std::vector<int64_t>> paddings = {{pad_h.first, pad_h.second},
                                                  {pad_w.first, pad_w.second}};
    builder::ConvDimensionNumbers dims_attr(
        /*input_batch_dimension=*/3,
        /*input_feature_dimension=*/0,
        /*input_spatial_dimensions=*/{1, 2},
        /*kernel_input_feature_dimension=*/0,
        /*kernel_output_feature_dimension=*/3,
        /*kernel_spatial_dimensions=*/{1, 2},
        /*output_batch_dimension=*/depthwise ? 3 : 2,
        /*output_feature_dimension=*/depthwise ? 2 : 3,
        /*output_spatial_dimensions=*/{0, 1});
    filter_grad = builder::Conv(input,
                                out_grad,
                                dims_attr,
                                /*window_strides=*/window_strides,
                                /*padding=*/paddings,
                                /*lhs_dilation=*/{1, 1},
                                /*rhs_dilation=*/rhs_dilation,
                                /*window_reversal=*/{},
                                /*auto_pad=*/"",
                                /*feature_group_count=*/1,
                                /*batch_group_count=*/depthwise ? group : 1,
                                /*precision_config=*/{});
    filter_grad.SetAttribute("op_type",
                             builder::Attribute("Conv2DBackpropFilter"));
    if (depthwise) {
      auto filter_grad_shape = filter_grad.GetType().GetShape();
      auto new_filter_grad_shape = filter_grad_shape;
      new_filter_grad_shape = {filter_grad_shape[0],
                               filter_grad_shape[1],
                               filter_grad_shape[3],
                               filter_grad_shape[2]};
      builder::Type new_type(new_filter_grad_shape,
                             filter_grad.GetType().GetPrimitiveType());
      filter_grad = builder::Reshape(filter_grad, new_type);
    }

    if (running_mode == RunningMode::ADAPTIVE) {
      // when run, it will do transpose for kernel in host, so here is not
      // insert transpose for transpose erase, record weight name
      // TransformUtil::TransAndRecordWeights(node, "Filter@GRAD",
      // filter_grad.GetType().GetShape());
    } else {
      filter_grad = builder::Transpose(filter_grad, {3, 2, 0, 1});
    }
  }
  // the first layer conv op maybe not need to calc input grad, so here return
  if (output_name_map.count("Input@GRAD") != 0 &&
      output_name_map["Input@GRAD"].size() == 0) {
    return std::make_shared<GcuOp>(filter_grad);
  }
  // calculate input_grad
  builder::Op input_grad;
  if (output_name_map.count("Input@GRAD") != 0 &&
      output_name_map["Input@GRAD"].size() > 0) {
    auto filter_reverse = builder::Reverse(filter, {0, 1}, filter.GetType());
    std::vector<int64_t> lhs_dilation = stride;
    std::vector<int64_t> rhs_dilation = dilation;
    auto pad_h = get_backprop_input_padding(
        ih, oh, kh, stride[0], dilation[0], padding[0]);
    auto pad_w = get_backprop_input_padding(
        iw, ow, kw, stride[1], dilation[1], padding[2]);
    std::vector<std::vector<int64_t>> paddings = {{pad_h.first, pad_h.second},
                                                  {pad_w.first, pad_w.second}};
    builder::ConvDimensionNumbers dims_attr(
        /*input_batch_dimension=*/0,
        /*input_feature_dimension=*/3,
        /*input_spatial_dimensions=*/{1, 2},
        /*kernel_input_feature_dimension=*/3,
        /*kernel_output_feature_dimension=*/2,
        /*kernel_spatial_dimensions=*/{0, 1},
        /*output_batch_dimension=*/0,
        /*output_feature_dimension=*/3,
        /*output_spatial_dimensions=*/{1, 2});
    input_grad = builder::Conv(out_grad,
                               filter_reverse,
                               dims_attr,
                               /*window_strides=*/{1, 1},
                               /*padding=*/paddings,
                               /*lhs_dilation=*/lhs_dilation,
                               /*rhs_dilation=*/rhs_dilation,
                               /*window_reversal=*/{},
                               /*auto_pad=*/"",
                               /*feature_group_count=*/depthwise ? group : 1,
                               /*batch_group_count=*/1,
                               /*precision_config=*/{});
    input_grad.SetAttribute("op_type",
                            builder::Attribute("Conv2DBackpropInput"));
    input_grad = builder::Transpose(input_grad, {0, 3, 1, 2});
  }
  std::vector<builder::Op> outputs;
  std::vector<std::string> output_names;
  if (output_name_map.count("Filter@GRAD") != 0 &&
      output_name_map["Filter@GRAD"].size() > 0) {
    output_names.push_back("Filter@GRAD");
    outputs.push_back(filter_grad);
  }
  if (output_name_map.count("Input@GRAD") != 0 &&
      output_name_map["Input@GRAD"].size() > 0) {
    output_names.push_back("Input@GRAD");
    outputs.push_back(input_grad);
  }
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

IMPLEMT_EQUIVALENCE_TRANS_FUNC(gcu_builder,
                               node,
                               map_inputs,
                               running_mode,
                               Conv2DTransposeEquivalenceTrans) {
  auto* op = node->Op();
  auto groups = PADDLE_GET_CONST(int, op->GetAttr("groups"));
  auto padding_algorithm =
      PADDLE_GET_CONST(std::string, op->GetAttr("padding_algorithm"));
  auto data_format = PADDLE_GET_CONST(std::string, op->GetAttr("data_format"));
  auto dilations = PADDLE_GET_CONST(std::vector<int>, op->GetAttr("dilations"));
  auto strides = PADDLE_GET_CONST(std::vector<int>, op->GetAttr("strides"));
  auto paddings = PADDLE_GET_CONST(std::vector<int>, op->GetAttr("paddings"));
  auto output_paddings =
      PADDLE_GET_CONST(std::vector<int>, op->GetAttr("output_padding"));
  auto output_size =
      PADDLE_GET_CONST(std::vector<int>, op->GetAttr("output_size"));
  std::vector<builder::Op> ops;
  // sorted_archetype_names :"Input", "Filter", "Bias"
  // necessary input
  if (map_inputs.count("Input") != 0) {
    VLOG(10) << "inputs size:" << map_inputs["Input"].size();
    auto op_ptr = map_inputs["Input"].at(0);
    if (data_format == "NCHW") {
      ops.emplace_back(builder::Transpose(*op_ptr, {0, 2, 3, 1}));
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
    auto transpose = builder::Transpose(*op_ptr, {2, 3, 1, 0});
    ops.emplace_back(builder::Reverse(transpose, {0, 1}, transpose.GetType()));
  } else {
    PADDLE_ENFORCE_EQ(
        true, false, platform::errors::NotFound("lack of [Filter] gcu op"));
  }
  // optional input
  if (map_inputs.count("Bias") != 0 && map_inputs["Bias"].size() != 0) {
    ops.push_back(*(map_inputs["Bias"].at(0)));
  }

  int64_t group = static_cast<int64_t>(groups);
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
  for (auto dim : paddings) {
    padding.emplace_back(static_cast<int64_t>(dim));
  }
  for (auto dim : output_paddings) {
    output_padding.emplace_back(static_cast<int64_t>(dim));
  }
  for (auto dim : output_size) {
    output_shape.emplace_back(static_cast<int64_t>(dim));
  }
  auto input_shape = ops[0].GetType().GetShape();
  auto kernel_shape = ops[1].GetType().GetShape();
  int64_t ih = input_shape[1];
  int64_t iw = input_shape[2];
  int64_t kh = kernel_shape[0];
  int64_t kw = kernel_shape[1];
  if (padding_algorithm == "SAME") {
    auto pad_h = get_same_padding_value(ih, kh, stride[0]);
    auto pad_w = get_same_padding_value(iw, kw, stride[1]);
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
  if (output_padding.size() == 0) {
    output_padding = {0, 0};
  } else if (output_padding.size() == 1) {
    output_padding = {output_padding[0], output_padding[0]};
  }
  auto oh = GetConvTransposeDim(ih,
                                kh,
                                stride[0],
                                dilation[0],
                                padding[0],
                                padding[1],
                                output_padding[0]);
  auto ow = GetConvTransposeDim(iw,
                                kw,
                                stride[1],
                                dilation[1],
                                padding[2],
                                padding[3],
                                output_padding[1]);
  auto real_padding = get_conv2d_transpose_padding({ih, iw},
                                                   {oh, ow},
                                                   {kh, kw},
                                                   stride,
                                                   dilation,
                                                   padding,
                                                   output_padding,
                                                   "NOTSET");
  builder::ConvDimensionNumbers dims_attr(/*input_batch_dimension=*/0,
                                          /*input_feature_dimension=*/3,
                                          /*input_spatial_dimensions=*/{1, 2},
                                          /*kernel_input_feature_dimension=*/3,
                                          /*kernel_output_feature_dimension=*/2,
                                          /*kernel_spatial_dimensions=*/{0, 1},
                                          /*output_batch_dimension=*/0,
                                          /*output_feature_dimension=*/3,
                                          /*output_spatial_dimensions=*/{1, 2});
  auto conv2d_transpose = builder::Conv(ops[0],
                                        ops[1],
                                        dims_attr,
                                        /*window_strides=*/{1, 1},
                                        /*padding=*/real_padding,
                                        /*lhs_dilation=*/stride,
                                        /*rhs_dilation=*/dilation,
                                        /*window_reversal=*/{},
                                        /*auto_pad=*/"",
                                        /*feature_group_count=*/group,
                                        /*batch_group_count=*/1,
                                        /*precision_config=*/{});
  conv2d_transpose.SetAttribute("op_type",
                                builder::Attribute("Conv2DBackpropInput"));
  if (data_format == "NCHW") {
    auto transpose = builder::Transpose(conv2d_transpose, {0, 3, 1, 2});
    return std::make_shared<GcuOp>(transpose);
  } else {
    return std::make_shared<GcuOp>(conv2d_transpose);
  }
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(gcu_builder,
                               node,
                               map_inputs,
                               running_mode,
                               Conv2DTransposeGradEquivalenceTrans) {
  // do not take bias in account because bias_grad will be calculated in
  // elementwise_add_grad input keys: Output@GRAD, Filter, Input output keys:
  // Filter@GRAD, Input@GRAD
  auto* op = node->Op();
  auto data_format = PADDLE_GET_CONST(std::string, op->GetAttr("data_format"));
  // here we only support layout of NCHW
  if (data_format != "NCHW") {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Only support NCHW for now, but got : %s", data_format.c_str()));
  }
  auto groups = PADDLE_GET_CONST(int, op->GetAttr("groups"));
  auto padding_algorithm =
      PADDLE_GET_CONST(std::string, op->GetAttr("padding_algorithm"));
  auto dilations_i32 =
      PADDLE_GET_CONST(std::vector<int>, op->GetAttr("dilations"));
  auto strides_i32 = PADDLE_GET_CONST(std::vector<int>, op->GetAttr("strides"));
  auto paddings_i32 =
      PADDLE_GET_CONST(std::vector<int>, op->GetAttr("paddings"));
  auto output_paddings_i32 =
      PADDLE_GET_CONST(std::vector<int>, op->GetAttr("output_padding"));
  int64_t group = static_cast<int64_t>(groups);
  std::vector<int64_t> stride;
  std::vector<int64_t> dilation;
  std::vector<int64_t> padding;
  std::vector<int64_t> output_padding;
  for (auto dim : strides_i32) {
    stride.emplace_back(static_cast<int64_t>(dim));
  }
  for (auto dim : dilations_i32) {
    dilation.emplace_back(static_cast<int64_t>(dim));
  }
  for (auto dim : paddings_i32) {
    padding.emplace_back(static_cast<int64_t>(dim));
  }
  for (auto dim : output_paddings_i32) {
    output_padding.emplace_back(static_cast<int64_t>(dim));
  }
  builder::Op out_grad =
      builder::Transpose(*(map_inputs["Output@GRAD"].at(0)), {0, 2, 3, 1});
  builder::Op input =
      builder::Transpose(*(map_inputs["Input"].at(0)), {0, 2, 3, 1});
  // filter layout is [ic, oc, kh, kw]
  builder::Op filter =
      builder::Transpose(*(map_inputs["Filter"].at(0)), {2, 3, 1, 0});
  auto input_shape = input.GetType().GetShape();
  auto kernel_shape = filter.GetType().GetShape();
  auto output_shape = out_grad.GetType().GetShape();
  int64_t ih = input_shape[1];
  int64_t iw = input_shape[2];
  int64_t kh = kernel_shape[0];
  int64_t kw = kernel_shape[1];
  int64_t oh = output_shape[1];
  int64_t ow = output_shape[2];
  bool depthwise = false;
  if (group > 1 && group == input_shape[3] && group == kernel_shape[3]) {
    depthwise = true;
  }
  if (groups > 1 && !depthwise) {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Only support groups == 1 or depthwise conv2d, but got : %d", groups));
  }
  if (padding_algorithm == "SAME") {
    auto pad_h = get_same_padding_value(ih, kh, stride[0]);
    auto pad_w = get_same_padding_value(iw, kw, stride[1]);
    padding = {pad_h[0], pad_h[1], pad_w[0], pad_w[1]};
  } else {
    if (padding.size() == 1) {
      padding = {padding[0], padding[0], padding[0], padding[0]};
    } else if (padding.size() == 2) {
      padding = {padding[0], padding[0], padding[1], padding[1]};
    } else if (padding.size() == 8) {
      padding = {padding[4], padding[5], padding[6], padding[7]};
    }
  }
  // print_shape(padding, "forward padding");
  auto output_name_map = op->Outputs();
  // calculate filter_grad
  builder::Op filter_grad;
  if (output_name_map.count("Filter@GRAD") != 0 &&
      output_name_map["Filter@GRAD"].size() > 0) {
    auto pad_h = get_backprop_filter_padding(
        oh, ih, kh, stride[0], dilation[0], padding[0], padding_algorithm);
    auto pad_w = get_backprop_filter_padding(
        ow, iw, kw, stride[1], dilation[1], padding[2], padding_algorithm);
    std::vector<std::vector<int64_t>> paddings = {{pad_h.first, pad_h.second},
                                                  {pad_w.first, pad_w.second}};
    builder::ConvDimensionNumbers dims_attr(
        /*input_batch_dimension=*/3,
        /*input_feature_dimension=*/0,
        /*input_spatial_dimensions=*/{1, 2},
        /*kernel_input_feature_dimension=*/0,
        /*kernel_output_feature_dimension=*/3,
        /*kernel_spatial_dimensions=*/{1, 2},
        /*output_batch_dimension=*/2,
        /*output_feature_dimension=*/3,
        /*output_spatial_dimensions=*/{0, 1});
    filter_grad = builder::Conv(out_grad,
                                input,
                                dims_attr,
                                /*window_strides=*/dilation,
                                /*padding=*/paddings,
                                /*lhs_dilation=*/{1, 1},
                                /*rhs_dilation=*/stride,
                                /*window_reversal=*/{},
                                /*auto_pad=*/"",
                                /*feature_group_count=*/1,
                                /*batch_group_count=*/1,
                                /*precision_config=*/{});
    filter_grad.SetAttribute("op_type",
                             builder::Attribute("Conv2DBackpropFilter"));
    filter_grad = builder::Transpose(filter_grad, {3, 2, 0, 1});
  }
  // calculate input_grad
  builder::Op input_grad;
  if (output_name_map.count("Input@GRAD") != 0 &&
      output_name_map["Input@GRAD"].size() > 0) {
    std::vector<int64_t> lhs_dilation = {1, 1};
    std::vector<int64_t> rhs_dilation = dilation;
    builder::ConvDimensionNumbers dims_attr(
        /*input_batch_dimension=*/0,
        /*input_feature_dimension=*/3,
        /*input_spatial_dimensions=*/{1, 2},
        /*kernel_input_feature_dimension=*/2,
        /*kernel_output_feature_dimension=*/3,
        /*kernel_spatial_dimensions=*/{0, 1},
        /*output_batch_dimension=*/0,
        /*output_feature_dimension=*/3,
        /*output_spatial_dimensions=*/{1, 2});
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
    input_grad.SetAttribute("op_type", builder::Attribute("Conv2DInference"));
    input_grad = builder::Transpose(input_grad, {0, 3, 1, 2});
  }
  std::vector<builder::Op> outputs;
  std::vector<std::string> output_names;
  if (output_name_map.count("Filter@GRAD") != 0 &&
      output_name_map["Filter@GRAD"].size() > 0) {
    output_names.push_back("Filter@GRAD");
    outputs.push_back(filter_grad);
  }
  if (output_name_map.count("Input@GRAD") != 0 &&
      output_name_map["Input@GRAD"].size() > 0) {
    output_names.push_back("Input@GRAD");
    outputs.push_back(input_grad);
  }
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

EQUIVALENCE_TRANS_FUNC_REG(kConv2D, INSENSITIVE, Conv2DEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kConv2DGrad,
                           INSENSITIVE,
                           Conv2DGradEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kDepthWiseConv2D,
                           INSENSITIVE,
                           Conv2DEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kDepthWiseConv2DGrad,
                           INSENSITIVE,
                           Conv2DGradEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kConv2DTranspose,
                           INSENSITIVE,
                           Conv2DTransposeEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kConv2DTransposeGrad,
                           INSENSITIVE,
                           Conv2DTransposeGradEquivalenceTrans);

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
