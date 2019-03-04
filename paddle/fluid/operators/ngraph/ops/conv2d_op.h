/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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
#include <unordered_map>
#include <vector>
#include "ngraph/ngraph.hpp"
#include "paddle/fluid/operators/ngraph/ops/op_bridge.h"
#include "paddle/fluid/platform/ngraph_helper.h"

namespace paddle {
namespace operators {
namespace ngraphs {

std::shared_ptr<ngraph::Node> GroupedConvolution(
    const std::shared_ptr<ngraph::Node>& data_batch,
    const std::shared_ptr<ngraph::Node>& filters, const ngraph::Strides strides,
    const ngraph::Strides dilations, const ngraph::CoordinateDiff& paddings,
    size_t groups) {
  auto& data_shape = data_batch->get_shape();
  auto& filter_shape = filters->get_shape();
  ngraph::NodeVector ng_slices;

  for (size_t i = 0; i < groups; ++i) {
    size_t channel_step = filter_shape.at(1);
    const std::vector<size_t> lower_bound{0, i * channel_step, 0, 0};
    const std::vector<size_t> upper_bound{data_shape.at(0),
                                          (i + 1) * channel_step,
                                          data_shape.at(2), data_shape.at(3)};
    auto data_slice = std::make_shared<ngraph::op::Slice>(
        data_batch, lower_bound, upper_bound);

    size_t filter_step = filter_shape.at(0) / groups;
    const std::vector<size_t> filter_lower_bound{i * filter_step, 0, 0, 0};
    const std::vector<size_t> filter_upper_bound{
        (i + 1) * filter_step, filter_shape.at(1), filter_shape.at(2),
        filter_shape.at(3)};
    auto filter_slice = std::make_shared<ngraph::op::Slice>(
        filters, filter_lower_bound, filter_upper_bound);
    auto ng_conv = std::make_shared<ngraph::op::Convolution>(
        data_slice, filter_slice, strides, dilations, paddings, paddings);
    ng_slices.push_back(ng_conv);
  }

  size_t concat_axis = 1;
  return std::make_shared<ngraph::op::Concat>(ng_slices, concat_axis);
}

std::shared_ptr<ngraph::Node> GroupedGradConvolutionFilter(
    const std::shared_ptr<ngraph::Node>& data_batch,
    const std::shared_ptr<ngraph::Node>& filters,
    const std::shared_ptr<ngraph::Node>& doutput, const ngraph::Strides strides,
    const ngraph::Strides dilations, const ngraph::CoordinateDiff& paddings,
    size_t groups) {
  auto& data_shape = data_batch->get_shape();
  auto& filter_shape = filters->get_shape();
  auto& out_shape = doutput->get_shape();
  ngraph::NodeVector ng_slices;

  for (size_t i = 0; i < groups; ++i) {
    size_t channel_step = filter_shape.at(1);
    const std::vector<size_t> lower_bound{0, i * channel_step, 0, 0};
    const std::vector<size_t> upper_bound{data_shape.at(0),
                                          (i + 1) * channel_step,
                                          data_shape.at(2), data_shape.at(3)};
    auto data_slice = std::make_shared<ngraph::op::Slice>(
        data_batch, lower_bound, upper_bound);

    size_t filter_step = data_shape.at(0);

    const std::vector<size_t> filter_lower_bound{i * filter_step, 0, 0, 0};
    const std::vector<size_t> filter_upper_bound{
        (i + 1) * filter_step, filter_shape.at(1), filter_shape.at(2),
        filter_shape.at(3)};
    auto filter_slice = std::make_shared<ngraph::op::Slice>(
        filters, filter_lower_bound, filter_upper_bound);

    const std::vector<size_t> olower_bound{0, i * filter_step, 0, 0};
    const std::vector<size_t> oupper_bound{out_shape.at(0),
                                           (i + 1) * filter_step,
                                           out_shape.at(2), out_shape.at(3)};
    auto out_slice = std::make_shared<ngraph::op::Slice>(doutput, olower_bound,
                                                         oupper_bound);

    auto ng_conv = std::make_shared<ngraph::op::ConvolutionBackpropFilters>(
        data_slice, filter_slice->get_shape(), out_slice, strides, dilations,
        paddings, paddings, ngraph::Strides{1, 1});

    ng_slices.push_back(ng_conv);
  }

  size_t concat_axis = 0;
  return std::make_shared<ngraph::op::Concat>(ng_slices, concat_axis);
}

std::shared_ptr<ngraph::Node> GroupedGradConvolutionData(
    const std::shared_ptr<ngraph::Node>& data_batch,
    const std::shared_ptr<ngraph::Node>& filters,
    const std::shared_ptr<ngraph::Node>& doutput, const ngraph::Strides strides,
    const ngraph::Strides dilations, const ngraph::CoordinateDiff& paddings,
    size_t groups) {
  auto& data_shape = data_batch->get_shape();
  auto& filter_shape = filters->get_shape();
  auto& out_shape = doutput->get_shape();
  ngraph::NodeVector ng_slices;

  for (size_t i = 0; i < groups; ++i) {
    size_t channel_step = filter_shape.at(1);
    const std::vector<size_t> lower_bound{0, i * channel_step, 0, 0};
    const std::vector<size_t> upper_bound{data_shape.at(0),
                                          (i + 1) * channel_step,
                                          data_shape.at(2), data_shape.at(3)};
    auto data_slice = std::make_shared<ngraph::op::Slice>(
        data_batch, lower_bound, upper_bound);

    size_t filter_step = data_shape.at(0);

    const std::vector<size_t> filter_lower_bound{i * filter_step, 0, 0, 0};
    const std::vector<size_t> filter_upper_bound{
        (i + 1) * filter_step, filter_shape.at(1), filter_shape.at(2),
        filter_shape.at(3)};
    auto filter_slice = std::make_shared<ngraph::op::Slice>(
        filters, filter_lower_bound, filter_upper_bound);

    const std::vector<size_t> olower_bound{0, i * filter_step, 0, 0};
    const std::vector<size_t> oupper_bound{out_shape.at(0),
                                           (i + 1) * filter_step,
                                           out_shape.at(2), out_shape.at(3)};
    auto out_slice = std::make_shared<ngraph::op::Slice>(doutput, olower_bound,
                                                         oupper_bound);

    auto ng_conv = std::make_shared<ngraph::op::ConvolutionBackpropData>(
        data_slice->get_shape(), filter_slice, out_slice, strides, dilations,
        paddings, paddings, ngraph::Strides{1, 1});
    ng_slices.push_back(ng_conv);
  }

  size_t concat_axis = 1;
  return std::make_shared<ngraph::op::Concat>(ng_slices, concat_axis);
}

void BuildConv2dNode(
    const std::shared_ptr<paddle::framework::OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto op_attrs = paddle::framework::AttrReader(op->Attrs());
  auto filters = paddle::platform::GetInputNode(op, "Filter", ngb_node_map);
  auto input = paddle::platform::GetInputNode(op, "Input", ngb_node_map);

  std::vector<int> strides = op_attrs.Get<std::vector<int>>("strides");
  std::vector<int> paddings = op_attrs.Get<std::vector<int>>("paddings");
  std::vector<int> dilations = op_attrs.Get<std::vector<int>>("dilations");

  const ngraph::Strides ng_strides{static_cast<size_t>(strides.at(0)),
                                   static_cast<size_t>(strides.at(1))};
  const ngraph::Strides ng_dilations{static_cast<size_t>(dilations.at(0)),
                                     static_cast<size_t>(dilations.at(1))};
  const ngraph::CoordinateDiff ng_paddings{
      static_cast<std::ptrdiff_t>(paddings.at(0)),
      static_cast<std::ptrdiff_t>(paddings.at(1))};

  int groups = static_cast<size_t>(op_attrs.Get<int>("groups"));
  PADDLE_ENFORCE_GE(groups, 1, "conv groups needs be no less than 1");

  std::shared_ptr<ngraph::Node> result;
  if (groups == 1) {
    result = std::make_shared<ngraph::op::Convolution>(
        input, filters, ng_strides, ng_dilations, ng_paddings, ng_paddings);
  } else {
    result = GroupedConvolution(input, filters, ng_strides, ng_dilations,
                                ng_paddings, groups);
  }
  paddle::platform::SetOutputNode(op, "Output", result, ngb_node_map);
}

void BuildConv2dGradNode(
    const std::shared_ptr<paddle::framework::OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto op_attrs = paddle::framework::AttrReader(op->Attrs());
  auto filter = paddle::platform::GetInputNode(op, "Filter", ngb_node_map);
  auto input = paddle::platform::GetInputNode(op, "Input", ngb_node_map);
  auto doutput =
      paddle::platform::GetInputNode(op, "Output@GRAD", ngb_node_map);

  int groups = op_attrs.Get<int>("groups");
  std::vector<int> strides = op_attrs.Get<std::vector<int>>("strides");
  std::vector<int> paddings = op_attrs.Get<std::vector<int>>("paddings");
  std::vector<int> dilations = op_attrs.Get<std::vector<int>>("dilations");

  const ngraph::Strides ng_strides{static_cast<size_t>(strides.at(0)),
                                   static_cast<size_t>(strides.at(1))};
  const ngraph::Strides ng_dilations{static_cast<size_t>(dilations.at(0)),
                                     static_cast<size_t>(dilations.at(1))};
  const ngraph::CoordinateDiff ng_paddings{
      static_cast<std::ptrdiff_t>(paddings.at(0)),
      static_cast<std::ptrdiff_t>(paddings.at(1))};

  std::shared_ptr<ngraph::Node> dfilter;
  std::shared_ptr<ngraph::Node> dinput;
  if (groups == 1) {
    dfilter = std::make_shared<ngraph::op::ConvolutionBackpropFilters>(
        input, filter->get_shape(), doutput, ng_strides, ng_dilations,
        ng_paddings, ng_paddings, ngraph::Strides{1, 1});

    dinput = std::make_shared<ngraph::op::ConvolutionBackpropData>(
        input->get_shape(), filter, doutput, ng_strides, ng_dilations,
        ng_paddings, ng_paddings, ngraph::Strides{1, 1});

  } else {
    dfilter = GroupedGradConvolutionFilter(input, filter, doutput, ng_strides,
                                           ng_dilations, ng_paddings, groups);
    dinput = GroupedGradConvolutionData(input, filter, doutput, ng_strides,
                                        ng_dilations, ng_paddings, groups);
  }

  paddle::platform::SetOutputNode(op, "Filter@GRAD", dfilter, ngb_node_map);
  paddle::platform::SetOutputNode(op, "Input@GRAD", dinput, ngb_node_map);
}
}  // namespace ngraphs
}  // namespace operators
}  // namespace paddle

REGISTER_NG_OP(conv2d, BuildConv2dNode);
REGISTER_NG_OP(conv2d_grad, BuildConv2dGradNode);
