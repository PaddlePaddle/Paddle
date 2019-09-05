/*Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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
#include <unordered_map>
#include <vector>
#include "ngraph/ngraph.hpp"
#include "paddle/fluid/operators/ngraph/ops/op_bridge.h"
#include "paddle/fluid/platform/ngraph_helper.h"

namespace paddle {
namespace operators {
namespace ngraphs {

void BuildSliceNode(
    const std::shared_ptr<framework::OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto input = paddle::platform::GetInputNode(op, "Input", ngb_node_map);
  auto input_shape = input->get_shape();
  auto op_attrs = framework::AttrReader(op->Attrs());
  auto axes = op_attrs.Get<std::vector<int>>("axes");
  auto starts = op_attrs.Get<std::vector<int>>("starts");
  auto ends = op_attrs.Get<std::vector<int>>("ends");
  ngraph::Coordinate ng_start, ng_end;
  int axis, start, end;
  for (size_t i = 0; i < input_shape.size(); ++i) {
    ng_start.push_back(0);
    ng_end.push_back(input_shape[i]);
  }
  for (size_t i = 0; i < axes.size(); ++i) {
    axis = input_shape[axes[i]];
    start = starts[i] < 0 ? (starts[i] + axis) : starts[i];
    end = ends[i] < 0 ? (ends[i] + axis) : ends[i];
    start = std::max(start, 0);
    end = std::max(end, 0);
    start = std::min(start, axis);
    end = std::min(end, axis);
    start = std::min(start, end);
    ng_start[axes[i]] = start;
    ng_end[axes[i]] = end;
  }
  auto out = std::make_shared<ngraph::op::Slice>(input, ng_start, ng_end);
  auto out_shape = out->get_shape();

  std::vector<size_t> out_axis_vec(out_shape.size());
  std::iota(out_axis_vec.begin(), out_axis_vec.end(), 0);

  paddle::platform::TrimTrailingSingularDims(&out_shape);
  auto out_dim = std::make_shared<ngraph::op::Reshape>(
      out, ngraph::AxisVector(out_axis_vec), ngraph::Shape(out_shape));

  platform::SetOutputNode(op, "Out", out_dim, ngb_node_map);
}

void BuildSliceGradNode(
    const std::shared_ptr<framework::OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto input = paddle::platform::GetInputNode(op, "Input", ngb_node_map);
  auto input_shape = input->get_shape();
  auto dout = paddle::platform::GetInputNode(op, "Out@GRAD", ngb_node_map);
  auto op_attrs = framework::AttrReader(op->Attrs());
  auto axes = op_attrs.Get<std::vector<int>>("axes");
  auto starts = op_attrs.Get<std::vector<int>>("starts");
  auto ends = op_attrs.Get<std::vector<int>>("ends");
  auto reshape = input_shape;
  ngraph::Coordinate ng_start, ng_end;
  int axis, start, end;
  for (size_t i = 0; i < input_shape.size(); ++i) {
    ng_start.push_back(0);
    ng_end.push_back(input_shape[i]);
  }
  for (size_t i = 0; i < axes.size(); ++i) {
    axis = input_shape[axes[i]];
    start = starts[i] < 0 ? (starts[i] + axis) : starts[i];
    end = ends[i] < 0 ? (ends[i] + axis) : ends[i];
    start = std::max(start, 0);
    end = std::max(end, 0);
    start = std::min(start, axis);
    end = std::min(end, axis);
    start = std::min(start, end);
    ng_start[axes[i]] = start;
    ng_end[axes[i]] = end;
    reshape[axes[i]] = end - start;
  }
  std::vector<size_t> axisVec(dout->get_shape().size());
  std::iota(axisVec.begin(), axisVec.end(), 0);
  auto dout_reshape = std::make_shared<ngraph::op::Reshape>(
      dout, ngraph::AxisVector(axisVec), reshape);

  std::shared_ptr<ngraph::Node> input0 = paddle::platform::CreateConstant(
      dout->get_element_type(), input_shape, {0});

  auto din = std::make_shared<ngraph::op::ReplaceSlice>(input0, dout_reshape,
                                                        ng_start, ng_end);
  platform::SetOutputNode(op, "Input@GRAD", din, ngb_node_map);
}
}  // namespace ngraphs
}  // namespace operators
}  // namespace paddle

REGISTER_NG_OP(slice, BuildSliceNode);
REGISTER_NG_OP(slice_grad, BuildSliceGradNode);
