/*Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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

void BuildPool2dNode(
    const std::shared_ptr<paddle::framework::OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto op_attrs = paddle::framework::AttrReader(op->Attrs());
  auto x = paddle::platform::GetInputNode(op, "X", ngb_node_map);
  auto x_shape = x->get_shape();

  std::string pooling_type = op_attrs.Get<std::string>("pooling_type");
  std::vector<int> ksize = op_attrs.Get<std::vector<int>>("ksize");
  std::vector<int> strides = op_attrs.Get<std::vector<int>>("strides");
  std::vector<int> paddings = op_attrs.Get<std::vector<int>>("paddings");

  PADDLE_ENFORCE_EQ(x_shape.size() - 2, ksize.size(),
                    "Handling 2d pooling only");

  if (op_attrs.Get<bool>("global_pooling")) {
    for (size_t i = 0; i < ksize.size(); ++i) {
      paddings[i] = 0;
      ksize[i] = static_cast<int>(x_shape.at(i + 2));
    }
  }

  ngraph::Shape ng_padding_below{static_cast<size_t>(paddings.at(0)),
                                 static_cast<size_t>(paddings.at(1))};
  ngraph::Shape ng_padding_above{static_cast<size_t>(paddings.at(0)),
                                 static_cast<size_t>(paddings.at(1))};
  ngraph::Shape ng_ksize_shape{static_cast<size_t>(ksize.at(0)),
                               static_cast<size_t>(ksize.at(1))};
  ngraph::Strides ng_strides{static_cast<size_t>(strides.at(0)),
                             static_cast<size_t>(strides.at(1))};

  auto ComputeCeiledOutput = [](size_t in, size_t k, size_t p, size_t s) {
    return (in - k + 2 * p) / s + 1;
  };

  if (op_attrs.Get<bool>("ceil_mode")) {
    auto dummy_out = paddle::platform::GetOutputNode(op, "Out", ngb_node_map);
    auto dummpy_shape = dummy_out->get_shape();
    for (size_t i = 0; i < ng_padding_above.size(); ++i) {
      auto desired_size = ComputeCeiledOutput(x_shape[i + 2], ksize[i],
                                              paddings[i], strides[i]);
      if (desired_size != dummpy_shape[i + 2]) {
        ng_padding_above[i] += strides[i];
      }
    }
  }

  bool padding_exclusive = op_attrs.Get<bool>("exclusive");
  if (pooling_type == "max") {
    auto pool2d = std::make_shared<ngraph::op::MaxPool>(
        x, ng_ksize_shape, ng_strides, ng_padding_below, ng_padding_above);
    paddle::platform::SetOutputNode(op, "Out", pool2d, ngb_node_map);
  } else if (pooling_type == "avg") {
    std::shared_ptr<ngraph::Node> pool2d;
    if (op_attrs.Get<bool>("adaptive")) {
      auto ComputeAdaptive = [](size_t in, size_t k) {
        return std::floor(in / k);
      };
      ng_strides[0] = x_shape.size() == 4
                          ? ComputeAdaptive(x_shape[3], ksize[0])
                          : ng_strides[0];
      ng_strides[1] = x_shape.size() == 4
                          ? ComputeAdaptive(x_shape[3], ksize[0])
                          : ng_strides[1];
      pool2d =
          std::make_shared<ngraph::op::AvgPool>(x, ng_ksize_shape, ng_strides);
    } else {
      pool2d = std::make_shared<ngraph::op::AvgPool>(
          x, ng_ksize_shape, ng_strides, ng_padding_below, ng_padding_above,
          !padding_exclusive);
    }
    paddle::platform::SetOutputNode(op, "Out", pool2d, ngb_node_map);
  } else {
    PADDLE_THROW("Support max and avg pooling only");
  }
}

void BuildPool2dGradNode(
    const std::shared_ptr<paddle::framework::OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto op_attrs = paddle::framework::AttrReader(op->Attrs());
  auto out = paddle::platform::GetInputNode(op, "Out", ngb_node_map);
  auto dout = paddle::platform::GetInputNode(op, "Out@GRAD", ngb_node_map);
  auto x = paddle::platform::GetInputNode(op, "X", ngb_node_map);
  auto x_shape = x->get_shape();

  std::string pooling_type = op_attrs.Get<std::string>("pooling_type");
  std::vector<int> ksize = op_attrs.Get<std::vector<int>>("ksize");
  std::vector<int> strides = op_attrs.Get<std::vector<int>>("strides");
  std::vector<int> paddings = op_attrs.Get<std::vector<int>>("paddings");

  PADDLE_ENFORCE_EQ(x_shape.size() - 2, ksize.size(),
                    "Handling 2d pooling only");

  if (op_attrs.Get<bool>("global_pooling")) {
    for (size_t i = 0; i < ksize.size(); ++i) {
      paddings[i] = 0;
      ksize[i] = static_cast<int>(x_shape.at(i + 2));
    }
  }

  ngraph::Shape ng_padding_below{static_cast<size_t>(paddings.at(0)),
                                 static_cast<size_t>(paddings.at(1))};
  ngraph::Shape ng_padding_above{static_cast<size_t>(paddings.at(0)),
                                 static_cast<size_t>(paddings.at(1))};
  ngraph::Shape ng_ksize_shape{static_cast<size_t>(ksize.at(0)),
                               static_cast<size_t>(ksize.at(1))};
  ngraph::Strides ng_strides{static_cast<size_t>(strides.at(0)),
                             static_cast<size_t>(strides.at(1))};

  bool padding_exclusive = op_attrs.Get<bool>("exclusive");
  if (pooling_type == "max") {
    auto pool2d_grad = std::make_shared<ngraph::op::MaxPoolBackprop>(
        x, dout, out, ng_ksize_shape, ng_strides, ng_padding_below,
        ng_padding_above);
    paddle::platform::SetOutputNode(op, "X@GRAD", pool2d_grad, ngb_node_map);
  } else if (pooling_type == "avg") {
    std::shared_ptr<ngraph::Node> pool2d_grad;
    if (op_attrs.Get<bool>("adaptive")) {
      auto ComputeAdaptive = [](size_t in, size_t k) {
        return std::floor(in / k);
      };
      ng_strides[0] = x_shape.size() == 4
                          ? ComputeAdaptive(x_shape[3], ksize[0])
                          : ng_strides[0];
      ng_strides[1] = x_shape.size() == 4
                          ? ComputeAdaptive(x_shape[3], ksize[0])
                          : ng_strides[1];
      pool2d_grad = std::make_shared<ngraph::op::AvgPoolBackprop>(
          x->get_shape(), dout, ng_ksize_shape, ng_strides, ng_padding_below,
          ng_padding_above, !padding_exclusive);
    } else {
      pool2d_grad = std::make_shared<ngraph::op::AvgPoolBackprop>(
          x->get_shape(), dout, ng_ksize_shape, ng_strides, ng_padding_below,
          ng_padding_above, !padding_exclusive);
    }
    paddle::platform::SetOutputNode(op, "X@GRAD", pool2d_grad, ngb_node_map);
  } else {
    PADDLE_THROW("Support max and avg pooling only");
  }
}
}  // namespace ngraphs
}  // namespace operators
}  // namespace paddle

REGISTER_NG_OP(pool2d, BuildPool2dNode);
REGISTER_NG_OP(pool2d_grad, BuildPool2dGradNode);
