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

#ifdef PADDLE_WITH_NGRAPH
#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "ngraph/ngraph.hpp"

namespace paddle {
namespace platform {

std::shared_ptr<ngraph::Node> Nhwc2Nchw(std::shared_ptr<ngraph::Node> in) {
  auto in_shape = in->get_shape();
  in_shape[0] = in->get_shape()[0];
  in_shape[1] = in->get_shape()[3];
  in_shape[2] = in->get_shape()[1];
  in_shape[3] = in->get_shape()[2];
  ngraph::AxisVector axis_vec = {0, 3, 1, 2};
  return std::make_shared<ngraph::op::Reshape>(in, axis_vec, in_shape);
}

std::shared_ptr<ngraph::Node> Nchw2Nhwc(std::shared_ptr<ngraph::Node> in) {
  auto in_shape = in->get_shape();
  in_shape[0] = in->get_shape()[0];
  in_shape[1] = in->get_shape()[2];
  in_shape[2] = in->get_shape()[3];
  in_shape[3] = in->get_shape()[1];
  ngraph::AxisVector axis_vec = {0, 2, 3, 1};
  return std::make_shared<ngraph::op::Reshape>(in, axis_vec, in_shape);
}

ngraph::Shape FlattenTo1d(ngraph::Shape sh, int num) {
  auto x1 = std::accumulate(std::begin(sh), std::end(sh) + num, 1,
                            std::multiplies<size_t>());
  size_t x1_l = (size_t)x1;
  return ngraph::Shape{x1_l};
}

ngraph::Shape FlattenTo2d(ngraph::Shape sh, int num) {
  auto x1 = std::accumulate(std::begin(sh), std::begin(sh) + num, 1,
                            std::multiplies<size_t>());
  auto x2 = std::accumulate(std::begin(sh) + num, std::end(sh), 1,
                            std::multiplies<size_t>());
  size_t x1_l = static_cast<size_t>(x1);
  size_t x2_l = static_cast<size_t>(x2);
  return ngraph::Shape{x1_l, x2_l};
}

std::shared_ptr<ngraph::Node> NgReshaper(std::shared_ptr<ngraph::Node> input,
                                         ngraph::Shape shape) {
  std::vector<size_t> input_order(input->get_shape().size());
  std::iota(std::begin(input_order), std::end(input_order), 0);
  return std::make_shared<ngraph::op::Reshape>(
      input, ngraph::AxisVector(input_order), shape);
}

std::shared_ptr<ngraph::Node> GetNode(
    const std::shared_ptr<paddle::framework::OperatorBase>& op,
    const std::string name, const paddle::framework::VariableNameMap& var_map,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto& var_names = var_map.at(name);
  if (var_names.size() == 0) return nullptr;
  if (ngb_node_map->find(var_names[0]) != ngb_node_map->end()) {
    return (*ngb_node_map)[var_names[0]];
  } else {
    return nullptr;
  }
}

std::shared_ptr<ngraph::Node> GetInputNode(
    const std::shared_ptr<paddle::framework::OperatorBase>& op,
    const std::string name,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  return GetNode(op, name, op->Inputs(), ngb_node_map);
}

std::shared_ptr<ngraph::Node> GetOutputNode(
    const std::shared_ptr<paddle::framework::OperatorBase>& op,
    const std::string name,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  return GetNode(op, name, op->Outputs(), ngb_node_map);
}

template <typename T>
std::shared_ptr<ngraph::Node> CreateConstant(const ngraph::element::Type& type,
                                             ngraph::Shape shape,
                                             std::initializer_list<T> values) {
  std::shared_ptr<ngraph::Node> result;
  if (values.size() == 1 && shape != ngraph::Shape{} &&  // NOLINT
      shape != ngraph::Shape{1}) {
    result = std::make_shared<ngraph::op::Constant>(type, ngraph::Shape{},
                                                    std::vector<T>{values});
    ngraph::AxisSet axis_set;
    for (size_t i = 0; i < shape.size(); ++i) axis_set.insert(i);
    result = std::make_shared<ngraph::op::Broadcast>(result, shape, axis_set);
  } else {
    result = std::make_shared<ngraph::op::Constant>(type, shape,
                                                    std::vector<T>{values});
  }
  return result;
}

void SetOutputNode(
    const std::shared_ptr<paddle::framework::OperatorBase>& op,
    const std::string name, std::shared_ptr<ngraph::Node> node,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto& var_names = op->Outputs().at(name);
  if (var_names.size() == 1) {
    (*ngb_node_map)[var_names[0]] = node;
  } else if (var_names.size() == 0) {
    (*ngb_node_map)[""] = node;
  } else {
    PADDLE_THROW("name %s has more than 1 var_names.", name);
  }
}

bool HasOutput(const std::shared_ptr<paddle::framework::OperatorBase>& op,
               const std::string name) {
  auto& outputs = op->Outputs();
  if (outputs.find(name) == outputs.end()) return false;
  return outputs.at(name).size() > 0;
}

inline void GetMidDims(const ngraph::Shape& x_shape,
                       const ngraph::Shape& y_shape, int axis, int* pre, int* n,
                       int* post) {
  *pre = 1;
  *n = 1;
  *post = 1;
  for (int i = 0; i < axis; ++i) {
    (*pre) *= x_shape[i];
  }

  for (size_t i = 0; i < y_shape.size(); ++i) {
    PADDLE_ENFORCE_EQ(x_shape[i + axis], y_shape[i],
                      "Broadcast dimension mismatch.");
    (*n) *= y_shape[i];
  }

  for (size_t i = axis + y_shape.size(); i < x_shape.size(); ++i) {
    (*post) *= x_shape[i];
  }
}

inline void TrimTrailingSingularDims(ngraph::Shape* shape) {
  // Remove trailing dimensions of size 1 for y
  auto actual_shape_size = shape->size();
  for (; actual_shape_size != 0; --actual_shape_size) {
    if ((*shape)[actual_shape_size - 1] != 1) {
      break;
    } else {
      shape->pop_back();
    }
  }
}

ngraph::element::Type GetNgType(paddle::framework::proto::VarType::Type dtype) {
  ngraph::element::Type ng_dtype;
  if (dtype == paddle::framework::proto::VarType::FP32) {
    ng_dtype = ngraph::element::f32;
  } else if (dtype == paddle::framework::proto::VarType::FP64) {
    ng_dtype = ngraph::element::f64;
  } else if (dtype == paddle::framework::proto::VarType::INT64) {
    ng_dtype = ngraph::element::i64;
  } else if (dtype == paddle::framework::proto::VarType::INT32) {
    ng_dtype = ngraph::element::i32;
  } else {
    PADDLE_THROW("unsupported data type: %s", dtype);
  }
  return ng_dtype;
}
}  // namespace platform
}  // namespace paddle

#endif
